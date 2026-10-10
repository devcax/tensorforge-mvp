import re
import unicodedata

REFERENCE_PATTERN = re.compile(r"\b[a-z]{3,5}-[a-z0-9]{5,}\b")
AMOUNT_PATTERN = re.compile(r"(?:lkr|rs\.?)\s*[\d,.]+k?|[\d,.]*\d[\d,.]*k?(?:\s*(?:lkr|rupees?))?")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.?!])\s+|\s+(?=agent:|customer:)")


def normalize_text(raw_text):
    if raw_text is None:
        return ""
    normalized = unicodedata.normalize("NFC", str(raw_text))
    return " ".join(normalized.split()).lower()


def build_transformer_input(channel, subject, text, language=None):
    language_tag = f"[{language}] " if language else ""
    return f"{language_tag}[{channel}] {normalize_text(subject)} {normalize_text(text)}".strip()


def build_detector_input(subject, text):
    return f"{normalize_text(subject)} {normalize_text(text)}".strip()


def build_body(subject, text):
    return f"{normalize_text(subject)} {normalize_text(text)}".strip()


def mask_entities(text):
    masked = REFERENCE_PATTERN.sub(" <ref> ", text)
    masked = AMOUNT_PATTERN.sub(" <amount> ", masked)
    return " ".join(masked.split())


def split_sentences(text):
    return [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]


def remove_filler(body, filler_sentences):
    kept_sentences = [sentence for sentence in split_sentences(body) if sentence not in filler_sentences]
    return " ".join(kept_sentences) if kept_sentences else body


def build_classic_input(channel, subject, text, filler_sentences=None):
    body = mask_entities(build_body(subject, text))
    if filler_sentences:
        body = remove_filler(body, filler_sentences)
    return f"[{channel}] {body}"


def build_clean_transformer_input(channel, subject, text, language, filler_sentences):
    clean_body = remove_filler(mask_entities(build_body(subject, text)), filler_sentences)
    language_tag = f"[{language}] " if language else ""
    return f"{language_tag}[{channel}] {clean_body}"