from .config import CATEGORY_TO_TEAM, HUMAN_REVIEW_THRESHOLD, NO_SECONDARY_LABEL, SPAM_CATEGORY


def clean_secondary(category, secondary):
    secondary_label = str(secondary)
    if category == SPAM_CATEGORY or secondary_label in (NO_SECONDARY_LABEL, category):
        return None
    if secondary_label not in CATEGORY_TO_TEAM:
        return None
    return secondary_label


def clean_urgency(category, is_urgent):
    return False if category == SPAM_CATEGORY else bool(is_urgent)


def clamp_confidence(confidence):
    return round(min(max(float(confidence), 0.0), 1.0), 4)


def build_prediction(ticket, category, confidence, secondary, is_urgent, model_version):
    category_label = str(category)
    confidence_value = clamp_confidence(confidence)
    prediction = {}
    if isinstance(ticket.get("ticket_id"), str):
        prediction["ticket_id"] = ticket["ticket_id"]
    prediction.update({
        "category": category_label,
        "secondary_category": clean_secondary(category_label, secondary),
        "team": CATEGORY_TO_TEAM[category_label],
        "is_urgent": clean_urgency(category_label, is_urgent),
        "confidence": confidence_value,
        "model_version": model_version,
        "needs_human_review": confidence_value < HUMAN_REVIEW_THRESHOLD,
    })
    return prediction
