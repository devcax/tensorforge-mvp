import json
import logging
import threading
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import (
    INFERENCE_BATCH_SIZE,
    MAX_TOKENS,
    TORCH_THREADS,
    URGENT_MODEL,
    URGENT_MODEL_CHOICES,
    USE_QUANTIZATION,
)
from .model_download import resolve_models_dir
from .preprocessing import (
    build_classic_input,
    build_clean_transformer_input,
    build_detector_input,
    build_transformer_input,
)
from .rules import build_prediction

logger = logging.getLogger("predictor")


def read_json_settings(file_path):
    if not file_path.exists():
        return {}
    with open(file_path, encoding="utf-8") as settings_file:
        return json.load(settings_file)


def read_json_setting(file_path, key, default):
    return float(read_json_settings(file_path).get(key, default))


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def quantize_if_enabled(model):
    if not USE_QUANTIZATION:
        return model
    try:
        return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception:
        logger.warning("quantization failed, using the full precision model")
        return model


def load_transformer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), dtype=torch.float32, local_files_only=True)
    model.eval()
    return tokenizer, quantize_if_enabled(model)


def softmax_rows(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


class TicketPredictor:
    def __init__(self, model_version, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir is not None else resolve_models_dir()
        self.category_transformer_dir = self.models_dir / "xlmr_category"
        self.urgent_transformer_dir = self.models_dir / "xlmr_urgent"
        torch.set_num_threads(TORCH_THREADS)
        self.model_version = model_version
        self.tokenizer_lock = threading.Lock()
        self.secondary_model = joblib.load(self.models_dir / "secondary_category_model.joblib")
        self.urgent_model = joblib.load(self.models_dir / "is_urgent_model.joblib")
        self.filler_sentences = joblib.load(self.models_dir / "filler_sentences.joblib")
        self.text_choice = joblib.load(self.models_dir / "text_choice.joblib")
        self.language_detector = self.load_language_detector()
        self.uses_transformer = self.category_transformer_dir.exists()
        if self.uses_transformer:
            self.load_category_transformer()
        else:
            self.classic_category_model = joblib.load(self.models_dir / "category_model.joblib")
        self.urgent_mode = self.choose_urgent_mode()
        if self.urgent_mode != "scratch":
            self.load_urgent_transformer()
        logger.info("urgency mode: %s", self.urgent_mode)

    def choose_urgent_mode(self):
        requested_mode = URGENT_MODEL if URGENT_MODEL in URGENT_MODEL_CHOICES else "ensemble"
        if requested_mode != URGENT_MODEL:
            logger.warning("unknown URGENT_MODEL '%s', using ensemble", URGENT_MODEL)
        if requested_mode != "scratch" and not self.urgent_transformer_dir.exists():
            logger.warning("%s is missing, using the from-scratch urgency model", self.urgent_transformer_dir)
            return "scratch"
        return requested_mode

    def load_language_detector(self):
        for model_dir in (self.category_transformer_dir, self.urgent_transformer_dir):
            detector_path = model_dir / "language_detector.joblib"
            if detector_path.exists():
                logger.info("language detector loaded from %s", model_dir.name)
                return joblib.load(detector_path)
        return None

    def load_category_transformer(self):
        self.tokenizer, self.category_model = load_transformer(self.category_transformer_dir)
        self.id_to_label = {int(index): label for index, label in self.category_model.config.id2label.items()}
        self.temperature = read_json_setting(self.category_transformer_dir / "calibration.json", "temperature", 1.0)
        logger.info("category transformer loaded, temperature %.3f", self.temperature)

    def load_urgent_transformer(self):
        self.urgent_tokenizer, self.urgent_transformer = load_transformer(self.urgent_transformer_dir)
        urgency_settings = read_json_settings(self.urgent_transformer_dir / "urgency.json")
        self.urgent_threshold = float(urgency_settings.get("threshold", 0.5))
        self.blend_threshold = float(urgency_settings.get("blend_threshold", self.urgent_threshold))
        self.urgent_input_variant = urgency_settings.get("input", "raw")
        logger.info(
            "urgency transformer loaded, threshold %.2f, blend threshold %.2f, input %s",
            self.urgent_threshold, self.blend_threshold, self.urgent_input_variant,
        )

    def detect_languages(self, tickets):
        if self.language_detector is None:
            return [None] * len(tickets)
        detector_inputs = [build_detector_input(ticket.get("subject", ""), ticket["text"]) for ticket in tickets]
        return [str(language) for language in self.language_detector.predict(detector_inputs)]

    def build_classic_inputs(self, tickets, text_variant):
        filler_sentences = self.filler_sentences if text_variant == "masked_nofiller" else None
        return [
            build_classic_input(ticket["channel"], ticket.get("subject", ""), ticket["text"], filler_sentences)
            for ticket in tickets
        ]

    def tokenize_batch(self, tokenizer, texts):
        with self.tokenizer_lock:
            return tokenizer(texts, truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors="pt")

    def run_transformer_batched(self, tokenizer, model, texts):
        with self.tokenizer_lock:
            token_counts = [len(ids) for ids in tokenizer(texts, truncation=True, max_length=MAX_TOKENS)["input_ids"]]
        order = sorted(range(len(texts)), key=lambda index: token_counts[index])
        all_logits = [None] * len(texts)
        for start in range(0, len(order), INFERENCE_BATCH_SIZE):
            chunk_indices = order[start:start + INFERENCE_BATCH_SIZE]
            encoded = self.tokenize_batch(tokenizer, [texts[index] for index in chunk_indices])
            with torch.inference_mode():
                chunk_logits = model(**encoded).logits.float()
            for position, index in enumerate(chunk_indices):
                all_logits[index] = chunk_logits[position]
        return torch.stack(all_logits)

    def build_category_transformer_input(self, ticket, language):
        return build_transformer_input(ticket["channel"], ticket.get("subject", ""), ticket["text"], language)

    def predict_category_batch(self, tickets, languages):
        texts = [self.build_category_transformer_input(ticket, language) for ticket, language in zip(tickets, languages)]
        logits = self.run_transformer_batched(self.tokenizer, self.category_model, texts)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        confidences, best_indices = probabilities.max(dim=-1)
        labels = [self.id_to_label[int(index)] for index in best_indices]
        return labels, [float(value) for value in confidences]

    def build_urgent_transformer_input(self, ticket, language):
        if self.urgent_input_variant == "masked_nofiller":
            return build_clean_transformer_input(
                ticket["channel"], ticket.get("subject", ""), ticket["text"], language, self.filler_sentences
            )
        return build_transformer_input(ticket["channel"], ticket.get("subject", ""), ticket["text"], language)

    def predict_urgent_probability_batch(self, tickets, languages):
        texts = [self.build_urgent_transformer_input(ticket, language) for ticket, language in zip(tickets, languages)]
        logits = self.run_transformer_batched(self.urgent_tokenizer, self.urgent_transformer, texts)
        return torch.softmax(logits, dim=-1)[:, 1].numpy().astype(float)

    def scratch_urgent_scores(self, tickets):
        classic_inputs = self.build_classic_inputs(tickets, self.text_choice["is_urgent"])
        scores = np.asarray(self.urgent_model.decision_function(classic_inputs), dtype=float)
        urgent_is_first_class = bool(self.urgent_model.classes_[0]) is True
        return -scores if urgent_is_first_class else scores

    def predict_categories_with_classic_model(self, tickets):
        classic_inputs = self.build_classic_inputs(tickets, self.text_choice["category"])
        probabilities = softmax_rows(np.asarray(self.classic_category_model.decision_function(classic_inputs)))
        best_indices = probabilities.argmax(axis=1)
        labels = self.classic_category_model.classes_[best_indices]
        confidences = probabilities[np.arange(len(best_indices)), best_indices]
        return [str(label) for label in labels], [float(value) for value in confidences]

    def predict_categories(self, tickets, languages=None):
        if not self.uses_transformer:
            return self.predict_categories_with_classic_model(tickets)
        if languages is None:
            languages = self.detect_languages(tickets)
        return self.predict_category_batch(tickets, languages)

    def predict_urgencies(self, tickets, languages):
        if self.urgent_mode == "scratch":
            return [bool(score > 0) for score in self.scratch_urgent_scores(tickets)]
        probabilities = self.predict_urgent_probability_batch(tickets, languages)
        if self.urgent_mode == "transformer":
            return [bool(probability >= self.urgent_threshold) for probability in probabilities]
        blended_scores = 0.5 * probabilities + 0.5 * sigmoid(self.scratch_urgent_scores(tickets))
        return [bool(score >= self.blend_threshold) for score in blended_scores]

    def predict_secondaries(self, tickets):
        return list(self.secondary_model.predict(
            self.build_classic_inputs(tickets, self.text_choice["secondary_category"])
        ))

    def predict_tickets(self, tickets):
        languages = self.detect_languages(tickets)
        categories, confidences = self.predict_categories(tickets, languages)
        secondaries = self.predict_secondaries(tickets)
        urgencies = self.predict_urgencies(tickets, languages)
        return [
            build_prediction(ticket, category, confidence, secondary, urgent, self.model_version)
            for ticket, category, confidence, secondary, urgent
            in zip(tickets, categories, confidences, secondaries, urgencies)
        ]
