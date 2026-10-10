import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "").strip()
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "main").strip() or "main"
HF_MODEL_CACHE_DIR = Path(os.getenv("HF_MODEL_CACHE_DIR", str(DATA_DIR / "huggingface")))
CATEGORY_TRANSFORMER_DIR = MODELS_DIR / "xlmr_category"
URGENT_TRANSFORMER_DIR = MODELS_DIR / "xlmr_urgent"
MODEL_VERSION_FILE = MODELS_DIR / "MODEL_VERSION"
JOB_DATABASE_PATH = DATA_DIR / "jobs.sqlite3"

TORCH_THREADS = int(os.getenv("TORCH_THREADS", "2"))
USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "0") == "1"
URGENT_MODEL = os.getenv("URGENT_MODEL", "ensemble").strip().lower()
URGENT_MODEL_CHOICES = ("scratch", "transformer", "ensemble")
HUMAN_REVIEW_THRESHOLD = float(os.getenv("HUMAN_REVIEW_THRESHOLD", "0.5"))
MAX_TOKENS = 384
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", "16"))
JOB_CHUNK_SIZE = 25
MODEL_WAIT_SECONDS = 90

PREDICT_MAX_BYTES = 1 * 1024 * 1024
BATCH_MAX_BYTES = 5 * 1024 * 1024
JOB_MAX_BYTES = 25 * 1024 * 1024
BATCH_MAX_TICKETS = 100
JOB_MAX_TICKETS = 5000
TEXT_MAX_LENGTH = 10000
SUBJECT_MAX_LENGTH = 500
TICKET_ID_MAX_LENGTH = 64
IDEMPOTENCY_KEY_MAX_LENGTH = 128
RESULTS_MAX_LIMIT = 5000

MAX_ACTIVE_JOBS = 4
JOB_RETENTION_HOURS = 24
RETRY_AFTER_SECONDS = 3

CHANNELS = ("email", "chat", "call_transcript")
SPAM_CATEGORY = "spam_irrelevant"
NO_SECONDARY_LABEL = "none"

CATEGORY_TO_TEAM = {
    "payment_refund": "Payments & Refunds",
    "ride_trip_issue": "Ride Operations",
    "lost_item": "Lost & Found",
    "order_missing_wrong": "Food Operations",
    "delivery_delay": "Delivery Operations",
    "food_quality": "Restaurant Quality",
    "account_promo": "Account Services",
    "safety_conduct": "Trust & Safety",
    "app_technical": "Tech Support",
    "general_inquiry": "Front-line Support",
    "spam_irrelevant": "Auto-close / Spam Filter",
}
