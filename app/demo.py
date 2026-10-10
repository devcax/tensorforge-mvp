import html
import json
import os
import time

import httpx

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from .demo_parts.batch import (
    INPUT_COLUMNS,
    SAMPLE_CSV,
    build_results_table,
    build_summary,
    call_batch_api,
    cleanup_export_file,
    export_results_csv,
    readable_label,
    parse_ticket_csv,
    predictions_json,
    render_problems,
    render_table,
)
from .demo_parts.layout import DEMO_CSS, PROBLEMS_CSS, SUMMARY_CSS, TABLE_CSS


def build_error_result(status_code, message, details=None):
    error = {"code": "demo_error", "message": message}
    if details is not None:
        error["details"] = details
    return {
        "success": False,
        "status_code": status_code,
        "payload": {"error": error},
        "request_time_ms": 0,
    }


def call_predict_api(api_key, channel, subject, ticket_text):
    if not api_key or not api_key.strip():
        return build_error_result(401, "API Key Missing", "Please provide a valid API key in the top header to proceed.")

    if not ticket_text or not ticket_text.strip():
        return build_error_result(
            422,
            "Invalid Ticket Text",
            [{"field": "text", "issue": "Ticket body must contain at least one non-whitespace character"}],
        )

    payload = {"channel": channel, "text": ticket_text.strip()}
    if subject and subject.strip():
        payload["subject"] = subject.strip()

    base_url = os.getenv("DEMO_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    started_at = time.perf_counter()

    try:
        response = httpx.post(
            f"{base_url}/predict",
            json=payload,
            headers={"X-API-Key": api_key.strip()},
            timeout=30.0,
        )
        request_time_ms = round((time.perf_counter() - started_at) * 1000)

        try:
            response_payload = response.json()
        except ValueError:
            response_payload = {"raw_response": response.text}

        return {
            "success": response.is_success,
            "status_code": response.status_code,
            "payload": response_payload,
            "request_time_ms": request_time_ms,
        }
    except httpx.TimeoutException:
        request_time_ms = round((time.perf_counter() - started_at) * 1000)
        return {
            "success": False,
            "status_code": 504,
            "payload": {
                "error": {
                    "code": "timeout",
                    "message": "Gateway Timeout",
                    "details": "The prediction request timed out after 30 seconds. The server may be under heavy load.",
                }
            },
            "request_time_ms": request_time_ms,
        }
    except httpx.ConnectError:
        request_time_ms = round((time.perf_counter() - started_at) * 1000)
        return {
            "success": False,
            "status_code": None,
            "payload": {
                "error": {
                    "code": "connection_refused",
                    "message": "Unable to Reach API Server",
                    "details": f"Could not connect to `{base_url}`. Please verify that your backend service is running.",
                }
            },
            "request_time_ms": request_time_ms,
        }
    except Exception as exc:
        request_time_ms = round((time.perf_counter() - started_at) * 1000)
        return {
            "success": False,
            "status_code": None,
            "payload": {
                "error": {
                    "code": "network_error",
                    "message": "Network Request Failed",
                    "details": str(exc),
                }
            },
            "request_time_ms": request_time_ms,
        }


def extract_error_info(payload, status_code):
    """Parses FastAPI, custom JSON, or plain text error responses into human-friendly fields."""
    message = None
    tip = None
    details = None

    if isinstance(payload, dict):
        if "error" in payload:
            err = payload["error"]
            if isinstance(err, dict):
                message = err.get("message")
                details = err.get("details")
            elif isinstance(err, str):
                message = err
        elif "detail" in payload:
            det = payload["detail"]
            if isinstance(det, str):
                message = det
            elif isinstance(det, list):
                message = "Validation Error"
                details = det
        elif "message" in payload:
            message = str(payload["message"])
            details = payload.get("details")
        elif "raw_response" in payload:
            raw = str(payload["raw_response"]).strip()
            if "<html" in raw.lower():
                message = f"Server returned an HTML error page (Status {status_code or 'Unknown'})"
            else:
                message = raw[:250] + ("..." if len(raw) > 250 else "")

    status_map = {
        400: ("Bad Request", "The syntax or format of the ticket payload is invalid."),
        401: ("Invalid API key", "Enter a valid API key in the header."),
        403: ("Access Forbidden", "You do not have permission to query this model endpoint."),
        404: ("Endpoint Not Found", "The route `/predict` was not found on the target host."),
        413: ("Payload Too Large", "The input text exceeds the maximum character threshold."),
        422: ("Unprocessable Entity", "One or more fields failed validation requirements."),
        429: ("Rate Limit Exceeded", "Too many requests. Please wait a moment before retrying."),
        500: ("Internal Server Error", "An unhandled exception occurred in the model worker."),
        502: ("Bad Gateway", "The upstream model container or proxy is offline."),
        503: ("Model Initializing", "The ML weights are currently loading into memory. Retry shortly."),
        504: ("Inference Timeout", "The classification request took too long to complete."),
    }

    if not message:
        if status_code in status_map:
            message, tip = status_map[status_code]
        elif status_code is None:
            message = "Connection Failed"
            tip = "Check whether your local backend service is running and accessible."
        else:
            message = f"HTTP Error {status_code}"
            tip = "The request could not be completed successfully."
    elif not tip and status_code in status_map:
        _, tip = status_map[status_code]

    return message, tip, details


def render_error_card(result):
    status_code = result.get("status_code")
    payload = result.get("payload", {})
    request_time_ms = result.get("request_time_ms", 0)

    title, tip, details = extract_error_info(payload, status_code)
    status_pill = f"HTTP {status_code}" if status_code is not None else "Network Error"

    details_content = ""
    if details is not None:
        if isinstance(details, list):
            items_html = ""
            for item in details:
                if isinstance(item, dict):
                    field_name = item.get("field") or " -> ".join(str(x) for x in item.get("loc", []))
                    issue_msg = item.get("issue") or item.get("msg") or json.dumps(item)
                    items_html += (
                        f'<li style="margin-bottom:6px"><strong>{html.escape(str(field_name))}:</strong> '
                        f'<span style="color:#7a271a">{html.escape(str(issue_msg))}</span></li>'
                    )
                else:
                    items_html += f"<li>{html.escape(str(item))}</li>"
            details_content = (
                '<div style="margin-top:14px;padding:12px 16px;background:#ffffff;'
                'border:1px solid #fecdca;border-radius:10px">'
                '<div style="font-size:12px;font-weight:700;color:#912018;text-transform:uppercase;margin-bottom:8px">'
                'Validation Issues</div>'
                f'<ul style="margin:0;padding-left:18px;font-size:13px;color:#344054">{items_html}</ul></div>'
            )
        else:
            details_str = details if isinstance(details, str) else json.dumps(details, indent=2)
            details_content = (
                '<pre style="margin-top:14px;padding:12px;border-radius:10px;background:#ffffff;'
                'border:1px solid #fecdca;font-size:12px;color:#475467;overflow-x:auto">'
                f"{html.escape(details_str)}</pre>"
            )

    tip_html = ""
    if tip:
        tip_html = (
            f'<div style="margin-top:8px;font-size:13px;color:#532219;line-height:1.4">'
            f'<span style="font-weight:600">Tip:</span> {html.escape(tip)}</div>'
        )

    return (
        '<div style="padding:18px;border:1px solid #fda29b;border-radius:8px;'
        'background:#fffbfa">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">'
        '<div style="display:flex;align-items:center;gap:8px">'
        f'<span style="font-size:16px;font-weight:700;color:#912018">{html.escape(title)}</span>'
        '</div>'
        f'<span style="padding:4px 8px;border-radius:6px;background:#fee4e2;color:#b42318;'
        f'font-size:12px;font-weight:700">{status_pill}</span>'
        '</div>'
        f"{tip_html}"
        f"{details_content}"
        '<div style="margin-top:16px;padding-top:12px;border-top:1px solid #fee4e2;'
        f'font-size:12px;color:#98a2b3">Request failed in {request_time_ms} ms</div>'
        '</div>'
    )


def render_empty_card():
    return (
        '<div style="min-height:270px;display:flex;flex-direction:column;justify-content:center;'
        'align-items:center;text-align:center;padding:24px;border:1px dashed #d0d5dd;border-radius:8px;'
        'background:#fafafa;color:#667085">'
        '<div style="font-size:16px;font-weight:600;color:#344054;margin-bottom:6px">Prediction preview area</div>'
        '<div style="font-size:13px;max-width:320px;line-height:1.5">'
        'Enter ticket details on the left and click <strong>Check</strong> to view category, assigned team, and urgency.'
        '</div></div>'
    )


def render_result_card(result):
    payload = result["payload"]
    category = html.escape(readable_label(payload.get("category", "Unknown")))
    team = html.escape(str(payload.get("team", "Unassigned")))
    secondary = payload.get("secondary_category")
    secondary_text = "None" if secondary is None else readable_label(secondary)
    secondary_text = html.escape(secondary_text)
    model_version = html.escape(str(payload.get("model_version", "1.0.0")))

    try:
        confidence = max(0.0, min(float(payload.get("confidence", 0)), 1.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence_percent = round(confidence * 100)
    urgent = payload.get("is_urgent") is True
    needs_review = payload.get("needs_human_review") is True

    # Color grading based on confidence score
    if confidence >= 0.8:
        conf_color = "#12b76a"
        conf_desc = "High confidence"
    elif confidence >= 0.5:
        conf_color = "#f79009"
        conf_desc = "Moderate confidence"
    else:
        conf_color = "#f04438"
        conf_desc = "Low confidence — review advised"

    urgency_badge = (
        '<span style="padding:5px 8px;border-radius:6px;background:#fef3f2;'
        'border:1px solid #fecdca;color:#b42318;font-size:12px;font-weight:600">URGENT</span>'
        if urgent
        else '<span style="padding:5px 8px;border-radius:6px;background:#f2f4f7;'
        'color:#475467;font-size:12px;font-weight:600">Standard Priority</span>'
    )

    review_banner = ""
    if needs_review:
        review_banner = (
            '<div style="margin-top:16px;padding:10px 12px;border-radius:8px;'
            'background:#fffaeb;border:1px solid #fedf89;color:#b54708;font-size:13px;'
            'font-weight:600;display:flex;align-items:center;gap:8px">'
            'Flagged for human review</div>'
        )

    request_time_ms = result.get("request_time_ms", 0)

    return (
        '<div style="min-height:270px;padding:20px;border:1px solid #eaecf0;border-radius:8px;'
        'background:#ffffff">'
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">'
        '<div>'
        '<div style="font-size:11px;font-weight:700;color:#667085;letter-spacing:0.05em;text-transform:uppercase">'
        'Predicted Category</div>'
        f'<div style="font-size:24px;font-weight:800;color:#101828;margin-top:4px">{category}</div>'
        '</div>'
        f"{urgency_badge}"
        '</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:20px;padding:14px;'
        'background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0">'
        '<div>'
        '<div style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase">Assigned Team</div>'
        f'<div style="font-size:15px;font-weight:700;color:#0f172a;margin-top:2px">{team}</div>'
        '</div>'
        '<div>'
        '<div style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase">Secondary Tag</div>'
        f'<div style="font-size:15px;font-weight:700;color:#0f172a;margin-top:2px">{secondary_text}</div>'
        '</div>'
        '</div>'
        '<div style="margin-top:20px">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
        f'<span style="font-size:13px;font-weight:600;color:#344054">Confidence: {conf_desc}</span>'
        f'<span style="font-size:14px;font-weight:700;color:#101828">{confidence_percent}%</span>'
        '</div>'
        '<div style="height:8px;border-radius:999px;background:#f2f4f7;overflow:hidden">'
        f'<div style="width:{confidence_percent}%;height:100%;background:{conf_color};border-radius:999px;'
        'transition:width 0.4s ease"></div>'
        '</div>'
        '</div>'
        f"{review_banner}"
        '<div style="margin-top:20px;padding-top:14px;border-top:1px solid #f2f4f7;'
        'display:flex;justify-content:space-between;align-items:center;font-size:12px;color:#98a2b3">'
        f'<span>Model {model_version}</span>'
        f'<span>{request_time_ms} ms</span>'
        '</div></div>'
    )


def render_output(result, view):
    formatted_visible = view == "Formatted"
    if result is None:
        formatted = render_empty_card()
        raw_json = ""
    elif result.get("success"):
        formatted = render_result_card(result)
        raw_json = json.dumps(result.get("payload", {}), ensure_ascii=False, indent=2)
    else:
        formatted = render_error_card(result)
        raw_json = json.dumps(result.get("payload", {}), ensure_ascii=False, indent=2)

    return (
        gr.update(value=formatted, visible=formatted_visible),
        gr.update(value=raw_json, visible=not formatted_visible),
    )


def build_theme():
    return gr.themes.Soft(
        primary_hue="orange",
        font=["system-ui", "sans-serif"],
        font_mono=["ui-monospace", "monospace"],
    )


def empty_batch_result(rows=None):
    return {"rows": rows or [], "predictions": [], "status": "", "problems": [], "running": False}


def load_batch_csv(file_path):
    upload = parse_ticket_csv(file_path)
    rows = upload["rows"]
    problems = upload["problems"]
    result = empty_batch_result(rows)
    status = f"{len(rows):,} tickets loaded, {len(problems):,} problems" if file_path else "Upload a CSV to begin."
    return (
        upload,
        render_table(INPUT_COLUMNS, rows),
        status,
        gr.update(value=render_problems(problems), visible=bool(problems)),
        gr.update(interactive=bool(rows) and not problems),
        result,
        gr.update(value=build_results_table([], [])),
        gr.update(value="[]"),
        build_summary([], len(rows)),
        "",
        gr.update(value=None, interactive=False),
        None,
    )


def render_batch_output(result, view):
    result = result or empty_batch_result()
    predictions = result["predictions"]
    return (
        gr.update(value=build_results_table(result["rows"], predictions), visible=view == "Table"),
        gr.update(value=predictions_json(predictions), visible=view == "JSON"),
    )


def classify_batch(api_key, upload):
    upload = upload or {"rows": [], "problems": []}
    rows = upload["rows"]
    for result in call_batch_api(api_key, rows):
        predictions = result["predictions"]
        running = result["running"]
        download_path = None
        if not running:
            try:
                download_path = export_results_csv(rows, predictions)
            except (OSError, ValueError):
                result["status"] += " (CSV export failed to write to disk)"
        yield (
            result,
            gr.update(value=build_results_table(rows, predictions)),
            gr.update(value=predictions_json(predictions)),
            build_summary(predictions, len(rows)),
            result["status"],
            gr.update(value=render_problems(result["problems"]), visible=bool(result["problems"])),
            gr.update(value=download_path, interactive=bool(download_path)),
            gr.update(interactive=not running and bool(rows) and not upload["problems"]),
            gr.update(interactive=not running),
            download_path,
        )


def build_demo():
    with gr.Blocks(
        title="RideEat Ticket Classifier",
        analytics_enabled=False,
        fill_height=True,
        fill_width=True,
        delete_cache=(3600, 3600),
    ) as demo:
        with gr.Column(elem_id="demo-shell"):
            with gr.Row(elem_id="demo-header"):
                gr.Markdown("# RideEat Ticket Classifier", elem_id="demo-title")
                api_key = gr.Textbox(
                    label="API key",
                    placeholder="Enter your API key",
                    type="password",
                    elem_id="demo-key",
                )

            with gr.Tabs(elem_id="demo-tabs"):
                with gr.Tab("Single Ticket", elem_classes="workspace-tab"):
                    with gr.Row(elem_classes="workspace-row"):
                        with gr.Column(scale=1, min_width=0, elem_classes="workspace-panel"):
                            channel = gr.Dropdown(
                                choices=[("Chat", "chat"), ("Email", "email"), ("Call transcript", "call_transcript")],
                                value="chat",
                                label="Communication Channel",
                            )
                            subject = gr.Textbox(
                                label="Subject (Optional)",
                                placeholder="Brief summary of the issue",
                            )
                            ticket_text = gr.Textbox(
                                label="Customer Message",
                                placeholder="Paste or type the customer request here...",
                                lines=6,
                                elem_classes="ticket-input",
                            )

                            check_button = gr.Button("Analyze Ticket", variant="primary", size="lg")

                        with gr.Column(scale=1, min_width=0, elem_classes="workspace-panel"):
                            single_view = gr.Radio(
                                choices=["Formatted", "JSON"],
                                value="Formatted",
                                label="View Format",
                            )
                            result_card = gr.HTML(
                                value=render_empty_card(),
                                elem_classes="single-output",
                                apply_default_css=False,
                            )
                            single_json = gr.Code(
                                label="Raw Response JSON",
                                language="json",
                                visible=False,
                                interactive=False,
                                lines=14,
                                elem_classes="json-view",
                            )

                with gr.Tab("Batch Processing (CSV)", elem_classes="workspace-tab"):
                    with gr.Row(elem_classes="workspace-row"):
                        with gr.Column(scale=1, min_width=0, elem_classes="workspace-panel"):
                            csv_upload = gr.File(
                                label="Upload Ticket CSV",
                                file_types=[".csv"],
                                type="filepath",
                                height=176,
                                elem_id="csv-upload",
                            )
                            gr.DownloadButton(
                                label="Download Sample Template",
                                value=str(SAMPLE_CSV) if SAMPLE_CSV.is_file() else None,
                                visible=SAMPLE_CSV.is_file(),
                                size="sm",
                                variant="secondary",
                            )
                            load_status = gr.Markdown("Upload a CSV file to inspect entries.", elem_classes="batch-status")
                            preview = gr.HTML(
                                value=render_table(INPUT_COLUMNS, []),
                                label="CSV Preview",
                                elem_classes="table-panel",
                                css_template=TABLE_CSS,
                                apply_default_css=False,
                            )
                            problems = gr.HTML(
                                visible=False,
                                elem_classes="problems-panel",
                                css_template=PROBLEMS_CSS,
                                apply_default_css=False,
                            )
                            classify_button = gr.Button("Classify All Tickets", variant="primary", interactive=False, size="lg")

                        with gr.Column(scale=1, min_width=0, elem_classes="workspace-panel"):
                            batch_view = gr.Radio(
                                choices=["Table", "JSON"],
                                value="Table",
                                label="Batch Output View",
                            )
                            summary = gr.HTML(
                                value=build_summary([], 0),
                                elem_classes="summary-panel",
                                css_template=SUMMARY_CSS,
                                apply_default_css=False,
                            )
                            batch_status = gr.Markdown("", elem_classes="batch-status")
                            results_table = gr.HTML(
                                value=build_results_table([], []),
                                label="Triage Results",
                                elem_classes="table-panel",
                                css_template=TABLE_CSS,
                                apply_default_css=False,
                            )
                            batch_json = gr.Code(
                                value="[]",
                                label="Predictions JSON",
                                language="json",
                                visible=False,
                                interactive=False,
                                lines=14,
                                elem_classes="json-view",
                            )
                            download = gr.DownloadButton("Download Classified CSV", interactive=False, variant="secondary")

        # States
        single_state = gr.State()
        upload_state = gr.State({"rows": [], "problems": []}, time_to_live=3600)
        batch_state = gr.State(empty_batch_result(), time_to_live=3600)
        export_path = gr.State()

        # Single Prediction Flow
        check_button.click(
            call_predict_api,
            inputs=[api_key, channel, subject, ticket_text],
            outputs=single_state,
            api_name=False,
        ).then(
            render_output,
            inputs=[single_state, single_view],
            outputs=[result_card, single_json],
            api_name=False,
            queue=False,
        )

        single_view.change(
            render_output,
            inputs=[single_state, single_view],
            outputs=[result_card, single_json],
            api_name=False,
            queue=False,
        )

        # Batch Processing Flow
        csv_upload.change(
            load_batch_csv,
            inputs=csv_upload,
            outputs=[
                upload_state,
                preview,
                load_status,
                problems,
                classify_button,
                batch_state,
                results_table,
                batch_json,
                summary,
                batch_status,
                download,
                export_path,
            ],
            api_name=False,
        )

        classify_button.click(
            classify_batch,
            inputs=[api_key, upload_state],
            outputs=[
                batch_state,
                results_table,
                batch_json,
                summary,
                batch_status,
                problems,
                download,
                classify_button,
                csv_upload,
                export_path,
            ],
            api_name=False,
            show_progress="minimal",
            trigger_mode="once",
        ).then(cleanup_export_file, inputs=export_path, api_name=False, queue=False)

        batch_view.change(
            render_batch_output,
            inputs=[batch_state, batch_view],
            outputs=[results_table, batch_json],
            api_name=False,
            queue=False,
        )

    return demo.queue(default_concurrency_limit=2)
