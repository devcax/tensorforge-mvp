# TensorForge 2.0: RideEat Ticket Classifier API

Classifies a customer support ticket into a primary `category`, an optional `secondary_category`
and an `is_urgent` flag, and routes it to a `team`. Follows `tensorforge-phase2-openapi-v2.yaml`.

## Models

| Target | Model | Input text |
|---|---|---|
| `category` | Fine-tuned `xlm-roberta-base` (multilingual), temperature-calibrated | channel + subject + text |
| `secondary_category` | TF-IDF (word + character n-grams) + LinearSVC, trained from scratch | amounts and reference codes masked |
| `is_urgent` | TF-IDF + LinearSVC, trained from scratch | masked, learned filler sentences removed |

Only `channel`, `subject` and `text` are used. `ticket_id` is never a feature.
With `HF_MODEL_REPO` configured, the service downloads the complete model bundle at startup
and reuses it from the persistent cache. Without it, local development uses `MODELS_DIR`
(default `models/`) and can fall back to the from-scratch category model.

## Model files

```
models/
├── MODEL_VERSION                      version string returned as model_version
├── xlmr_category/                     fine-tuned transformer (config, weights, tokenizer, calibration.json)
├── category_model.joblib              from-scratch fallback for category
├── secondary_category_model.joblib
├── is_urgent_model.joblib
├── filler_sentences.joblib
└── text_choice.joblib
```

Bump `models/MODEL_VERSION` whenever any model file changes.

## Host models on Hugging Face and code on GitHub

The `models/` directory is excluded from Git and the Docker build. Keep your local model
files until you have uploaded them. GitHub already rejected the initial push because the
two weights files exceed its 100 MiB limit.

### 1. Upload the complete model bundle

Activate your Python environment and install `requirements.txt` as described below. Your
chosen model repository is the public `devcax/rideat-models`:

```powershell
hf auth login
hf repos create devcax/rideat-models --repo-type model --public --exist-ok
hf upload devcax/rideat-models ./models . --repo-type model --exclude ".cache/**"
```

Use a token with write access for the upload. The upload puts the **contents** of `models/` at the
repository root: `MODEL_VERSION`, all five root `.joblib` files, `xlmr_category/`, and
`xlmr_urgent/`. Upload the tokenizer JSON files, calibration, language detector and urgency
settings along with the weights; the application needs the whole bundle.

If using the Python API, use `upload_folder(folder_path="./models", path_in_repo="",
repo_id="devcax/rideat-models", repo_type="model")`. Do not upload the project root (`"."`),
which also contains application files and your local `.env`.

### 2. Configure downloads

Add these settings to your existing `.env` (keep your existing `API_KEY`):

```dotenv
HF_MODEL_REPO=devcax/rideat-models
HF_MODEL_REVISION=main
```

Your public repository needs no token for downloads. If you later make it private, set
`HF_TOKEN` to a deployment token with read access to that model repository. Keep tokens in
`.env` or your hosting platform's secret settings. `.env.example` contains no credentials.

For production, replace `main` with the full model repository commit hash shown on Hugging
Face after the upload. This pins all model files to one version. Change that hash to deploy
updated models; a cached branch such as `main` is deliberately reused without checking for
updates on every restart.

Startup first checks the cache for a complete bundle. If absent or incomplete, it downloads
the required files before loading the predictor. A failed download leaves the service
unready and logs the error; it does not silently switch to a different model. Restart after
fixing credentials or connectivity; existing downloaded files can be reused.

The cache defaults to `data/huggingface/` and can be changed with `HF_MODEL_CACHE_DIR`.
Docker Compose stores it in the existing `job_data` volume alongside the job database, so
container replacement preserves both. Do not use `docker compose down -v` if you want to
keep that data. The first download is roughly 1.1 GB; the health check allows 30 minutes
for the initial download and loading. Later starts reuse the cache and can run offline.

### 3. Fix the rejected GitHub push

Your current local history contains one initial commit. Run these commands from the project
root to remove models from that commit **while keeping the files on disk**, include the
download code, and retry the push:

```powershell
git rm -r --cached models
git add .gitignore .dockerignore Dockerfile docker-compose.yml requirements.txt README.md .env.example app
git commit --amend --no-edit
git push -u origin main
```

Adding `.gitignore` or deleting models in a later commit is insufficient because the large
files would remain in earlier history. The amend commands above apply to this single,
unpushed initial commit. If more commits are added before cleanup, remove the large files
from every affected commit instead (for example with `git filter-repo`).

### 4. Start the application

For local development, use the Uvicorn command below. For Docker, use the Docker commands
below after setting `HF_MODEL_REPO` in `.env`. Inspect progress with:

```powershell
docker compose logs -f api
```

`GET /health` returns `503` during download/loading and `200` when ready. Open `/demo`
after readiness. Hugging Face stores the model files; inference runs on your own API server.

## Install locally

Use Python 3.12. `requirements.txt` contains the pinned runtime dependencies for the API,
model inference, and Gradio demo. Install the CPU build of PyTorch first to avoid installing
GPU packages on the production CPU server:

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate`
on Linux/macOS), then run:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --env-file .env
```

Set `API_KEY` in `.env` before starting. `python-dotenv` enables Uvicorn's `--env-file` option.
The model bundle includes `tokenizer.json` and uses the Tokenizers backend, so `sentencepiece`
is not needed. Scikit-learn is required to load the saved `.joblib` classifiers even though the
API does not import it directly. Packages required by these libraries are installed automatically.

## Run with Docker

```bash
docker build -t tensorforge-api .
docker run -d --name tensorforge-api --restart unless-stopped --env-file .env -p 127.0.0.1:8000:8000 -v tensorforge-job-data:/service/data tensorforge-api
```

The image installs CPU PyTorch, runs as a regular user, and includes a readiness health check.
Keep one Uvicorn worker because the model and SQLite job worker are managed in one process.
Set `HF_MODEL_REPO` and, for private models, `HF_TOKEN` in `.env` before starting.
Persist `/service/data` to preserve downloaded models and job results across container replacement.

For the existing Caddy deployment, also set `DOMAIN` in `.env` to your server's hostname and
point its DNS record to the server. Then run:

```bash
docker compose config --quiet
docker compose up --build -d
docker compose ps
```

Compose waits for the API to become healthy before starting Caddy. The API port binds to
localhost; Caddy exposes ports 80 and 443 and forwards requests using `Caddyfile`.

The first model download, dependency installation, and Caddy's public TLS certificate setup
require internet access. Model inference and the demo can run offline once the configured
model revision is fully cached. `GET /health` returns `503` during download/loading, then `200`.

## Demo

Open `http://localhost:8000/demo`, enter the API key, choose a channel, and paste a ticket to classify it.

The key in the header is shared by the **Single ticket** and **Batch (CSV)** tabs. Batch uploads require
`ticket_id, channel, subject, text` columns and accept up to 5,000 tickets. Download a sample from the tab
or use `demo/sample_tickets.csv`. All rows are validated before classification, then sent in chunks of 100.
Results arrive as each chunk completes; switch between Table and JSON or download a CSV for Excel.
If a later chunk fails, completed results remain available to view and download.

The desktop layout keeps scrolling inside the tables, JSON panels, and ticket text area. Mobile screens stack
the panels. `DEMO_API_BASE_URL` defaults to `http://127.0.0.1:8000`; set it if the service listens on another port.

Optional environment variables: `TORCH_THREADS` (default 2), `USE_QUANTIZATION` (default 0),
`HUMAN_REVIEW_THRESHOLD` (default 0.5).

## Confidence and human review

`confidence` is the softmax probability of the predicted category after temperature scaling.
The temperature was fitted on the validation set to minimise negative log-likelihood, so the
value tracks how often the model is actually right.

`needs_human_review` is `true` when `confidence` is below **0.5**. Below that point the model is
more likely wrong than right about the primary category, so the ticket goes to a human triage
queue instead of being auto-routed. See `scripts/evaluate_validation.py` for the share of tickets
flagged and the accuracy on auto-routed tickets.

## Consistency rules

- `secondary_category` is `null` when it would equal `category`.
- `spam_irrelevant` always has `is_urgent: false` and `secondary_category: null`.
- `team` comes from the fixed category table.

## Batch jobs

Jobs are stored in SQLite (`data/jobs.sqlite3`) and processed by one background worker, so
`/health` and `/predict` stay responsive. One running job plus three queued jobs are accepted;
more return `429`. Results are kept for 24 hours, then return `410`. If the service restarts
mid-job, the job is marked `failed` with `error.code = interrupted`. Each ticket is classified on
its own, so predictions never depend on batch position or neighbours.

## Testing

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python scripts/check_api.py --api-key <key> --schemas-dir path/to/schemas
python scripts/evaluate_validation.py --validation-csv path/to/validation.csv
```

`check_api.py` validates every response against the official JSON Schemas and tests auth,
error codes, batch ordering, job flow, idempotency, paging and deletion.
