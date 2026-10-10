# TensorForge: RideEat Ticket Classifier

Classifies support tickets by category and urgency, and routes them to the appropriate team.

**Demo:** [tensorforge.devcax.com/demo](https://tensorforge.devcax.com/demo)

## Models

| Prediction | Model |
| --- | --- |
| Primary category | Fine-tuned multilingual XLM-RoBERTa |
| Secondary category | TF-IDF + LinearSVC |
| Urgency | XLM-RoBERTa + TF-IDF/LinearSVC ensemble |

## Model files

The model bundle is hosted on [Hugging Face](https://huggingface.co/devcax/rideat-models).
Its repository contains:

```text
MODEL_VERSION
category_model.joblib
secondary_category_model.joblib
is_urgent_model.joblib
filler_sentences.joblib
text_choice.joblib
xlmr_category/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    calibration.json
    language_detector.joblib
xlmr_urgent/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    urgency.json
```

## Hugging Face configuration

Set these values in `.env`:

```dotenv
HF_MODEL_REPO=devcax/rideat-models
HF_MODEL_REVISION=main
```

The repository is public, so no Hugging Face token is required. The first startup downloads
approximately 1.1 GB of model files. Later starts reuse the cache in the Docker data volume.
To use a specific model version, replace `main` with its Hugging Face commit hash.

## Run with Docker

Install Git and Docker with Docker Compose, then clone the project:

```bash
git clone https://github.com/devcax/tensorforge-mvp.git
cd tensorforge-mvp
cp .env.example .env
```

Edit `.env`: set `API_KEY` to your own secret value and keep the Hugging Face settings above.
`API_KEY` protects the classification API and is the key you enter in the demo.

Build and start the API:

```bash
docker compose up --build -d --remove-orphans
docker compose logs -f api
```

Wait for the models to download and load, then open [localhost:8000/demo](http://localhost:8000/demo).
`GET /health` returns `200` when ready. Downloaded models and batch jobs persist in the data volume.

The API binds to `127.0.0.1:8000` on the host. For remote access, configure your own proxy
to forward requests to that address.
