import html
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import httpx
import pandas as pd

from ..validation import find_ticket_issues

INPUT_COLUMNS = ["ticket_id", "channel", "subject", "text"]
PREDICTION_COLUMNS = [
    "category", "secondary_category", "team", "is_urgent", "confidence",
    "model_version", "needs_human_review",
]
RESULT_COLUMNS = [
    "ticket_id", "channel", "text", "category", "team", "secondary_category",
    "is_urgent", "confidence", "needs_human_review",
]
SAMPLE_CSV = Path(__file__).resolve().parents[2] / "demo" / "sample_tickets.csv"
MAX_ROWS = 5000
CHUNK_SIZE = 100
COLUMN_LABELS = {
    "ticket_id": "Ticket ID", "channel": "Channel", "subject": "Subject", "text": "Text",
    "category": "Category", "team": "Team", "secondary_category": "Secondary category",
    "is_urgent": "Urgent", "confidence": "Confidence", "needs_human_review": "Needs review",
    "model_version": "Model version",
}


def validate_ticket_rows(rows):
    problems = []
    seen_ids = set()
    for index, ticket in enumerate(rows):
        row_number = index + 2
        issues = find_ticket_issues(ticket, ticket_id_required=True)
        ticket_id = ticket.get("ticket_id", "")
        if not ticket_id.strip():
            issues.append({"field": "ticket_id", "issue": "must not be empty"})
        if ticket_id in seen_ids:
            issues.append({"field": "ticket_id", "issue": "duplicate of an earlier row"})
        seen_ids.add(ticket_id)
        problems.extend({"row": row_number, **issue} for issue in issues)
    return problems


def parse_ticket_csv(file_path):
    if not file_path:
        return {"rows": [], "problems": []}
    try:
        frame = pd.read_csv(
            file_path, dtype=str, keep_default_na=False, encoding="utf-8-sig", nrows=MAX_ROWS + 1,
        )
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError, ValueError):
        return {"rows": [], "problems": [{"issue": "Could not read this file. Upload a valid UTF-8 CSV."}]}

    frame.columns = [column.strip().lower() for column in frame.columns]
    missing_columns = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing_columns:
        return {"rows": [], "problems": [{"issue": "Missing required columns: " + ", ".join(missing_columns)}]}
    if any(list(frame.columns).count(column) > 1 for column in INPUT_COLUMNS):
        return {"rows": [], "problems": [{"issue": "Required column names must be unique."}]}
    if len(frame) > MAX_ROWS:
        return {"rows": [], "problems": [{"issue": "The CSV exceeds the 5,000-ticket limit."}]}
    if frame.empty:
        return {"rows": [], "problems": [{"issue": "The CSV contains no tickets."}]}

    rows = frame[INPUT_COLUMNS].to_dict(orient="records")
    return {"rows": rows, "problems": validate_ticket_rows(rows)}


def truncate_text(value, length=80):
    text = " ".join(str(value).split())
    return text if len(text) <= length else text[:length - 1] + "…"


def readable_label(value):
    return str(value).replace("_", " ").capitalize()


def render_table(columns, rows):
    headings = "".join(f"<th scope='col'>{html.escape(COLUMN_LABELS.get(column, readable_label(column)))}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            if column == "is_urgent" and value is True:
                content = '<span class="urgent">URGENT</span>'
            elif column == "confidence" and isinstance(value, (int, float)):
                content = f"{value * 100:.1f}%"
            elif isinstance(value, bool):
                content = "Yes" if value else "No"
            elif column in {"channel", "category", "secondary_category"} and value is not None:
                content = html.escape(truncate_text(readable_label(value)))
            else:
                content = html.escape(truncate_text("None" if value is None else value))
            cells.append(f"<td>{content}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    if not body:
        body.append(f'<tr><td colspan="{len(columns)}" class="empty">No tickets to display yet.</td></tr>')
    return f'<div class="table-scroll"><table><thead><tr>{headings}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def build_results_table(rows, predictions):
    tickets = {ticket["ticket_id"]: ticket for ticket in rows}
    results = [{**tickets.get(prediction["ticket_id"], {}), **prediction} for prediction in predictions]
    return render_table(RESULT_COLUMNS, results)


def build_summary(predictions, total):
    urgent_count = sum(prediction.get("is_urgent") is True for prediction in predictions)
    review_count = sum(prediction.get("needs_human_review") is True for prediction in predictions)
    categories = Counter(prediction.get("category", "Unknown") for prediction in predictions)
    top_categories = " · ".join(
        f"{html.escape(readable_label(category))}: {count:,}" for category, count in categories.most_common(3)
    )
    return (
        '<div class="summary-counts">'
        f'<span><strong>{total:,}</strong> total tickets</span>'
        f'<span><strong>{len(predictions):,}</strong> classified</span>'
        f'<span><strong>{urgent_count:,}</strong> urgent</span>'
        f'<span><strong>{review_count:,}</strong> need review</span></div>'
        f'<div class="summary-categories">{top_categories or "Categories appear after classification."}</div>'
    )


def render_problems(problems):
    if not problems:
        return ""
    items = []
    for problem in problems:
        row_label = f"CSV row {problem['row']}: " if problem.get("row") is not None else ""
        field_label = f"{problem['field']} — " if problem.get("field") else ""
        text = row_label + field_label + str(problem.get("issue", "Invalid value"))
        items.append(f"<li>{html.escape(text)}</li>")
    return '<strong>Problems</strong><ul>' + "".join(items) + "</ul>"


def split_into_chunks(rows):
    for start in range(0, len(rows), CHUNK_SIZE):
        yield start, rows[start:start + CHUNK_SIZE]


def build_batch_error(status_code, payload, chunk_start):
    messages = {
        401: "Invalid API key.",
        413: "This chunk is too large for the API.",
        415: "Unsupported request content type.",
        422: "Invalid tickets. Check the Problems list.",
        503: "Model is still loading, try again.",
    }
    message = messages.get(status_code, "The batch request could not be completed.")
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    details = error.get("details", []) if isinstance(error, dict) else []
    problems = []
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            index = detail.get("index")
            row = chunk_start + index + 2 if isinstance(index, int) and 0 <= index < CHUNK_SIZE else None
            problems.append({"row": row, "field": detail.get("field", ""), "issue": detail.get("issue", "Invalid value")})
    return message, problems


def call_batch_api(api_key, rows):
    predictions = []
    problems = validate_ticket_rows(rows)
    state = {"rows": rows, "predictions": [], "problems": problems, "running": False, "key_status": "Key not checked"}
    if not rows or len(rows) > MAX_ROWS or problems:
        yield {**state, "status": "Upload a valid CSV with 1 to 5,000 tickets first."}
        return
    if not api_key or not api_key.strip():
        yield {**state, "status": "Invalid API key. Enter a key in the header.", "key_status": "Invalid API key"}
        return

    base_url = os.getenv("DEMO_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    chunk_count = math.ceil(len(rows) / CHUNK_SIZE)
    for chunk_number, (start, chunk) in enumerate(split_into_chunks(rows), start=1):
        progress = f"Chunk {chunk_number} of {chunk_count}, {len(predictions):,} / {len(rows):,} tickets"
        yield {**state, "predictions": predictions.copy(), "running": True, "status": progress}
        try:
            response = httpx.post(
                f"{base_url}/predict/batch", json={"tickets": chunk}, headers={"X-API-Key": api_key}, timeout=300,
            )
            payload = response.json()
        except (httpx.HTTPError, httpx.InvalidURL):
            yield {**state, "predictions": predictions.copy(), "status": "Could not reach the API."}
            return
        except (ValueError, TypeError):
            yield {**state, "predictions": predictions.copy(), "status": "The API returned an unreadable response."}
            return

        state["key_status"] = "Invalid API key" if response.status_code == 401 else "Key OK"
        if not response.is_success:
            message, issues = build_batch_error(response.status_code, payload, start)
            yield {**state, "predictions": predictions.copy(), "status": message, "problems": issues}
            return

        chunk_predictions = payload.get("predictions") if isinstance(payload, dict) else None
        returned_ids = [item.get("ticket_id") if isinstance(item, dict) else None for item in chunk_predictions] if isinstance(chunk_predictions, list) else []
        if returned_ids != [ticket["ticket_id"] for ticket in chunk]:
            yield {**state, "predictions": predictions.copy(), "status": "The API returned incomplete or out-of-order results."}
            return

        predictions.extend(chunk_predictions)
        complete = chunk_number == chunk_count
        progress = f"Chunk {chunk_number} of {chunk_count}, {len(predictions):,} / {len(rows):,} tickets"
        yield {**state, "predictions": predictions.copy(), "running": not complete, "status": progress + (" — complete" if complete else "")}


def export_results_csv(rows, predictions):
    if not predictions:
        return None
    tickets = {ticket["ticket_id"]: ticket for ticket in rows}
    results = [{**tickets[prediction["ticket_id"]], **prediction} for prediction in predictions]
    with tempfile.NamedTemporaryFile(prefix="rideeat-results-", suffix=".csv", delete=False) as output:
        file_path = output.name
    pd.DataFrame(results, columns=INPUT_COLUMNS + PREDICTION_COLUMNS).to_csv(file_path, index=False, encoding="utf-8-sig")
    return file_path


def cleanup_export_file(file_path):
    if file_path:
        path = Path(file_path)
        if path.name.startswith("rideeat-results-") and path.parent.resolve() == Path(tempfile.gettempdir()).resolve():
            path.unlink(missing_ok=True)


def predictions_json(predictions):
    return json.dumps(predictions, ensure_ascii=False, indent=2)
