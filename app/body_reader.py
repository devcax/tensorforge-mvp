import json

from .errors import ApiError


def ensure_json_content_type(request):
    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";")[0].strip().lower()
    if media_type != "application/json":
        raise ApiError(415, "unsupported_media_type", "Content-Type must be application/json.")


def build_payload_too_large_error():
    return ApiError(413, "payload_too_large", "Request body exceeds the maximum allowed size.")


async def read_limited_body(request, max_bytes):
    declared_length = request.headers.get("content-length", "")
    if declared_length.isdigit() and int(declared_length) > max_bytes:
        raise build_payload_too_large_error()
    chunks = []
    received_bytes = 0
    async for chunk in request.stream():
        received_bytes += len(chunk)
        if received_bytes > max_bytes:
            raise build_payload_too_large_error()
        chunks.append(chunk)
    return b"".join(chunks)


def parse_json(raw_body):
    try:
        return json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ApiError(400, "malformed_json", "Request body is not valid JSON.")


async def read_json_body(request, max_bytes):
    ensure_json_content_type(request)
    raw_body = await read_limited_body(request, max_bytes)
    return parse_json(raw_body)
