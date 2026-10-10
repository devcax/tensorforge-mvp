import re

from .config import CHANNELS, SUBJECT_MAX_LENGTH, TEXT_MAX_LENGTH, TICKET_ID_MAX_LENGTH
from .errors import validation_failed

NON_WHITESPACE_PATTERN = re.compile(r"\S")


def build_issue(field, issue):
    return {"field": field, "issue": issue}


def check_channel(ticket):
    channel = ticket.get("channel")
    if channel is None:
        return [build_issue("channel", "is required")]
    if not isinstance(channel, str) or channel not in CHANNELS:
        return [build_issue("channel", "must be one of email, chat, call_transcript")]
    return []


def check_text(ticket):
    text = ticket.get("text")
    if text is None:
        return [build_issue("text", "is required")]
    if not isinstance(text, str):
        return [build_issue("text", "must be a string")]
    if len(text) > TEXT_MAX_LENGTH:
        return [build_issue("text", f"must be at most {TEXT_MAX_LENGTH} characters")]
    if not NON_WHITESPACE_PATTERN.search(text):
        return [build_issue("text", "must contain at least one non-whitespace character")]
    return []


def check_subject(ticket):
    if "subject" not in ticket:
        return []
    subject = ticket["subject"]
    if not isinstance(subject, str):
        return [build_issue("subject", "must be a string")]
    if len(subject) > SUBJECT_MAX_LENGTH:
        return [build_issue("subject", f"must be at most {SUBJECT_MAX_LENGTH} characters")]
    return []


def check_ticket_id(ticket, is_required):
    if "ticket_id" not in ticket:
        return [build_issue("ticket_id", "is required")] if is_required else []
    ticket_id = ticket["ticket_id"]
    if not isinstance(ticket_id, str):
        return [build_issue("ticket_id", "must be a string")]
    if len(ticket_id) > TICKET_ID_MAX_LENGTH:
        return [build_issue("ticket_id", f"must be at most {TICKET_ID_MAX_LENGTH} characters")]
    return []


def find_ticket_issues(ticket, ticket_id_required):
    if not isinstance(ticket, dict):
        return [build_issue("ticket", "must be a JSON object")]
    return (
        check_channel(ticket)
        + check_text(ticket)
        + check_subject(ticket)
        + check_ticket_id(ticket, ticket_id_required)
    )


def validate_single_ticket(body):
    issues = find_ticket_issues(body, ticket_id_required=False)
    if issues:
        raise validation_failed(issues)
    return body


def find_duplicate_id_issues(tickets):
    duplicate_issues = []
    seen_ids = set()
    for index, ticket in enumerate(tickets):
        ticket_id = ticket.get("ticket_id") if isinstance(ticket, dict) else None
        if not isinstance(ticket_id, str):
            continue
        if ticket_id in seen_ids:
            duplicate_issues.append({"index": index, "field": "ticket_id", "issue": "duplicate of an earlier item"})
        seen_ids.add(ticket_id)
    return duplicate_issues


def validate_ticket_list(body, max_tickets):
    if not isinstance(body, dict) or "tickets" not in body:
        raise validation_failed([build_issue("tickets", "is required")])
    tickets = body["tickets"]
    if not isinstance(tickets, list) or not 1 <= len(tickets) <= max_tickets:
        raise validation_failed([build_issue("tickets", f"must contain between 1 and {max_tickets} items")])
    item_issues = []
    for index, ticket in enumerate(tickets):
        for issue in find_ticket_issues(ticket, ticket_id_required=True):
            item_issues.append({"index": index, **issue})
    all_issues = item_issues + find_duplicate_id_issues(tickets)
    if all_issues:
        all_issues.sort(key=lambda issue: issue["index"])
        raise validation_failed(all_issues)
    return tickets
