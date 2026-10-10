import hmac
import os

from .errors import ApiError

EXPECTED_API_KEY = os.getenv("API_KEY", "")


def collect_provided_keys(request):
    provided_keys = []
    header_key = request.headers.get("x-api-key")
    if header_key:
        provided_keys.append(header_key)
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        provided_keys.append(token.strip())
    return provided_keys


def keys_match(provided_key, expected_key):
    return hmac.compare_digest(provided_key.encode("utf-8"), expected_key.encode("utf-8"))


def build_unauthorized_error(message):
    return ApiError(401, "unauthorized", message, headers={"WWW-Authenticate": "Bearer"})


def require_api_key(request):
    if not EXPECTED_API_KEY:
        raise build_unauthorized_error("Service API key is not configured.")
    provided_keys = collect_provided_keys(request)
    if not provided_keys:
        raise build_unauthorized_error("Missing API key. Send X-API-Key or Authorization Bearer.")
    if not any(keys_match(provided_key, EXPECTED_API_KEY) for provided_key in provided_keys):
        raise build_unauthorized_error("Invalid API key.")
