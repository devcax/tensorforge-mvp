import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from .config import HF_MODEL_CACHE_DIR, HF_MODEL_REPO, HF_MODEL_REVISION, MODELS_DIR

logger = logging.getLogger("model_download")

MODEL_FILES = (
    "MODEL_VERSION",
    "category_model.joblib",
    "secondary_category_model.joblib",
    "is_urgent_model.joblib",
    "filler_sentences.joblib",
    "text_choice.joblib",
    "xlmr_category/config.json",
    "xlmr_category/model.safetensors",
    "xlmr_category/tokenizer.json",
    "xlmr_category/tokenizer_config.json",
    "xlmr_category/calibration.json",
    "xlmr_category/language_detector.joblib",
    "xlmr_urgent/config.json",
    "xlmr_urgent/model.safetensors",
    "xlmr_urgent/tokenizer.json",
    "xlmr_urgent/tokenizer_config.json",
    "xlmr_urgent/urgency.json",
)


def missing_model_files(model_dir):
    return [
        name for name in MODEL_FILES
        if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0
    ]


def resolve_models_dir():
    if not HF_MODEL_REPO:
        if not MODELS_DIR.is_dir():
            raise RuntimeError("Set HF_MODEL_REPO to download models, or provide a local MODELS_DIR.")
        return MODELS_DIR

    download_options = {
        "repo_id": HF_MODEL_REPO,
        "repo_type": "model",
        "revision": HF_MODEL_REVISION,
        "cache_dir": str(HF_MODEL_CACHE_DIR),
        "allow_patterns": list(MODEL_FILES),
    }
    try:
        cached_dir = Path(snapshot_download(**download_options, local_files_only=True))
    except LocalEntryNotFoundError:
        cached_dir = None
    if cached_dir is not None and not missing_model_files(cached_dir):
        logger.info("using cached models from %s at %s", HF_MODEL_REPO, HF_MODEL_REVISION)
        return cached_dir

    logger.info("downloading models from %s at %s", HF_MODEL_REPO, HF_MODEL_REVISION)
    model_dir = Path(snapshot_download(**download_options))
    missing = missing_model_files(model_dir)
    if missing:
        raise RuntimeError("Hugging Face model repository is missing required files: " + ", ".join(missing))
    logger.info("model download complete")
    return model_dir
