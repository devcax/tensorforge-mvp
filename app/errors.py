from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status_code, code, message, details=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        self.headers = headers


def build_error_response(status_code, code, message, details=None, headers=None):
    error_body = {"code": code, "message": message}
    if details:
        error_body["details"] = details
    return JSONResponse(status_code=status_code, content={"error": error_body}, headers=headers)


def validation_failed(details):
    return ApiError(422, "validation_error", "Request failed validation.", details=details)
