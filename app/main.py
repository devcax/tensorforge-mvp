import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .auth import require_api_key
from .body_reader import read_json_body
from .config import (
    BATCH_MAX_BYTES,
    BATCH_MAX_TICKETS,
    IDEMPOTENCY_KEY_MAX_LENGTH,
    JOB_CHUNK_SIZE,
    JOB_DATABASE_PATH,
    JOB_MAX_BYTES,
    JOB_MAX_TICKETS,
    MAX_ACTIVE_JOBS,
    MODEL_VERSION_FILE,
    MODEL_WAIT_SECONDS,
    PREDICT_MAX_BYTES,
    RESULTS_MAX_LIMIT,
    RETRY_AFTER_SECONDS,
)
from .errors import ApiError, build_error_response, validation_failed
from .jobs import JobStore, JobWorker, build_status_body, is_job_expired
from .model_download import resolve_models_dir
from .validation import validate_single_ticket, validate_ticket_list

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("service")

RETRY_AFTER_HEADER = {"Retry-After": str(RETRY_AFTER_SECONDS)}


def read_model_version(version_file=MODEL_VERSION_FILE):
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").strip() or "v1.0"
    return "v1.0"


class ServiceState:
    def __init__(self):
        self.model_version = read_model_version()
        self.predictor = None
        self.ready_event = threading.Event()
        self.job_store = None
        self.job_worker = None


service_state = ServiceState()


def load_predictor():
    try:
        models_dir = resolve_models_dir()
        service_state.model_version = read_model_version(models_dir / "MODEL_VERSION")
        from .predictor import TicketPredictor
        service_state.predictor = TicketPredictor(service_state.model_version, models_dir=models_dir)
        service_state.ready_event.set()
        logger.info("model loaded, version %s", service_state.model_version)
    except Exception:
        logger.exception("model loading failed")


def get_ready_predictor():
    return service_state.predictor if service_state.ready_event.is_set() else None


@asynccontextmanager
async def lifespan(application):
    service_state.job_store = JobStore(JOB_DATABASE_PATH)
    service_state.job_store.mark_interrupted_jobs()
    threading.Thread(target=load_predictor, daemon=True, name="model-loader").start()
    service_state.job_worker = JobWorker(service_state.job_store, get_ready_predictor, JOB_CHUNK_SIZE)
    service_state.job_worker.start()
    yield


app = FastAPI(title="TensorForge Ticket Classifier", lifespan=lifespan)


@app.middleware("http")
async def echo_request_id(request: Request, call_next):
    response = await call_next(request)
    request_id = request.headers.get("x-request-id")
    if request_id:
        response.headers["X-Request-ID"] = request_id[:128]
    return response


@app.exception_handler(ApiError)
async def handle_api_error(request: Request, error: ApiError):
    return build_error_response(error.status_code, error.code, error.message, error.details, error.headers)


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(request: Request, error: StarletteHTTPException):
    known_errors = {
        404: ("not_found", "Route not found."),
        405: ("method_not_allowed", "Method not allowed for this route."),
    }
    code, message = known_errors.get(error.status_code, ("http_error", str(error.detail)))
    return build_error_response(error.status_code, code, message, headers=getattr(error, "headers", None))


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, error: Exception):
    logger.exception("unexpected error")
    return build_error_response(500, "internal_error", "Unexpected server error.")


async def wait_for_predictor():
    if not service_state.ready_event.is_set():
        await run_in_threadpool(service_state.ready_event.wait, MODEL_WAIT_SECONDS)
    if not service_state.ready_event.is_set():
        raise ApiError(503, "model_loading", "Model is still loading. Retry shortly.", headers=RETRY_AFTER_HEADER)
    return service_state.predictor


def read_idempotency_key(request):
    idempotency_key = request.headers.get("idempotency-key")
    if idempotency_key is not None and len(idempotency_key) > IDEMPOTENCY_KEY_MAX_LENGTH:
        raise validation_failed([{"field": "Idempotency-Key", "issue": "must be at most 128 characters"}])
    return idempotency_key or None


def parse_query_integer(request, name, default, minimum, maximum):
    raw_value = request.query_params.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        raise validation_failed([{"field": name, "issue": "must be an integer"}])
    if value < minimum or (maximum is not None and value > maximum):
        allowed_range = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise validation_failed([{"field": name, "issue": f"must be {allowed_range}"}])
    return value


def find_job_or_raise(job_id):
    job = service_state.job_store.get_job(job_id)
    if job is None:
        raise ApiError(404, "job_not_found", "No job with this id.")
    if is_job_expired(job):
        raise ApiError(410, "job_expired", "Job results have expired.")
    return job


@app.get("/health")
async def health():
    if not service_state.ready_event.is_set():
        return JSONResponse(status_code=503, content={"status": "loading", "model_version": None, "model_loaded": False})
    return JSONResponse(content={"status": "ok", "model_version": service_state.model_version, "model_loaded": True})


@app.post("/predict")
async def predict(request: Request):
    require_api_key(request)
    body = await read_json_body(request, PREDICT_MAX_BYTES)
    ticket = validate_single_ticket(body)
    predictor = await wait_for_predictor()
    predictions = await run_in_threadpool(predictor.predict_tickets, [ticket])
    return JSONResponse(content=predictions[0])


@app.post("/predict/batch")
async def predict_batch(request: Request):
    require_api_key(request)
    body = await read_json_body(request, BATCH_MAX_BYTES)
    tickets = validate_ticket_list(body, BATCH_MAX_TICKETS)
    predictor = await wait_for_predictor()
    started_at = time.perf_counter()
    predictions = await run_in_threadpool(predictor.predict_tickets, tickets)
    meta = {
        "count": len(predictions),
        "model_version": service_state.model_version,
        "processing_time_ms": int((time.perf_counter() - started_at) * 1000),
    }
    return JSONResponse(content={"predictions": predictions, "meta": meta})


@app.post("/batch/jobs")
async def submit_batch_job(request: Request):
    require_api_key(request)
    body = await read_json_body(request, JOB_MAX_BYTES)
    tickets = validate_ticket_list(body, JOB_MAX_TICKETS)
    idempotency_key = read_idempotency_key(request)
    outcome, job = await run_in_threadpool(
        service_state.job_store.submit_job, tickets, idempotency_key, service_state.model_version, MAX_ACTIVE_JOBS
    )
    if outcome == "full":
        raise ApiError(429, "too_many_jobs", "Job queue is full. Retry later.", headers=RETRY_AFTER_HEADER)
    service_state.job_worker.wake()
    headers = {"Location": f"/batch/jobs/{job['job_id']}", **RETRY_AFTER_HEADER}
    return JSONResponse(status_code=202, content=build_status_body(job), headers=headers)


@app.get("/batch/jobs/{job_id}")
async def get_batch_job(job_id: str, request: Request):
    require_api_key(request)
    job = find_job_or_raise(job_id)
    headers = RETRY_AFTER_HEADER if job["status"] in ("queued", "running") else None
    return JSONResponse(content=build_status_body(job), headers=headers)


@app.delete("/batch/jobs/{job_id}")
async def delete_batch_job(job_id: str, request: Request):
    require_api_key(request)
    if not service_state.job_store.delete_job(job_id):
        raise ApiError(404, "job_not_found", "No job with this id.")
    return Response(status_code=204)


@app.get("/batch/jobs/{job_id}/results")
async def get_batch_job_results(job_id: str, request: Request):
    require_api_key(request)
    job = find_job_or_raise(job_id)
    if job["status"] != "succeeded":
        raise ApiError(
            409, "job_not_ready", f"Job status is '{job['status']}'. Results are available once it is 'succeeded'."
        )
    offset = parse_query_integer(request, "offset", 0, 0, None)
    limit = parse_query_integer(request, "limit", None, 1, RESULTS_MAX_LIMIT)
    predictions = await run_in_threadpool(service_state.job_store.load_results, job_id)
    page_limit = limit if limit is not None else max(len(predictions), 1)
    page_end = offset + page_limit
    return JSONResponse(content={
        "job_id": job_id,
        "status": "succeeded",
        "total": len(predictions),
        "offset": offset,
        "limit": page_limit,
        "next_offset": page_end if page_end < len(predictions) else None,
        "model_version": job["model_version"],
        "predictions": predictions[offset:page_end],
    })


import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from .demo import DEMO_CSS, build_demo, build_theme

app = gr.mount_gradio_app(
    app, build_demo(), path="/", theme=build_theme(), css=DEMO_CSS,
    app_kwargs={"exception_handlers": {StarletteHTTPException: handle_http_error}},
    footer_links=[], run_history=False, show_error=False,
)
