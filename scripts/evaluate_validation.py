import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.predictor import TicketPredictor


def load_validation(csv_path):
    dataframe = pd.read_csv(csv_path, encoding="utf-8", dtype=str, keep_default_na=False)
    dataframe["is_urgent"] = dataframe["is_urgent"].str.strip().str.lower() == "true"
    dataframe["secondary_category"] = dataframe["secondary_category"].str.strip().replace({"": "none"})
    return dataframe


def build_tickets(dataframe):
    return [
        {"ticket_id": row["ticket_id"], "channel": row["channel"], "subject": row["subject"], "text": row["text"]}
        for _, row in dataframe.iterrows()
    ]


def expected_calibration_error(confidences, correct, bin_count=10):
    bin_edges = np.linspace(0, 1, bin_count + 1)
    total_error = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        if in_bin.any():
            total_error += in_bin.mean() * abs(confidences[in_bin].mean() - correct[in_bin].mean())
    return round(float(total_error), 4)


def score_predictions(dataframe, predictions):
    predicted_categories = np.array([p["category"] for p in predictions])
    predicted_secondaries = np.array([p["secondary_category"] or "none" for p in predictions])
    predicted_urgent = np.array([p["is_urgent"] for p in predictions])
    confidences = np.array([p["confidence"] for p in predictions])
    review_flags = np.array([p["needs_human_review"] for p in predictions])
    true_categories = dataframe["category"].to_numpy()
    correct = (predicted_categories == true_categories).astype(float)
    return {
        "category_macro_f1": round(f1_score(true_categories, predicted_categories, average="macro"), 4),
        "category_accuracy": round(accuracy_score(true_categories, predicted_categories), 4),
        "secondary_macro_f1": round(f1_score(dataframe["secondary_category"], predicted_secondaries, average="macro"), 4),
        "urgent_f1": round(f1_score(dataframe["is_urgent"], predicted_urgent), 4),
        "expected_calibration_error": expected_calibration_error(confidences, correct),
        "share_sent_to_human_review": round(float(review_flags.mean()), 4),
        "accuracy_when_auto_routed": round(float(correct[~review_flags].mean()), 4) if (~review_flags).any() else None,
    }


def score_by_language(dataframe, predictions):
    predicted_categories = np.array([p["category"] for p in predictions])
    language_scores = {}
    for language in sorted(dataframe["language"].unique()):
        mask = (dataframe["language"] == language).to_numpy()
        language_scores[language] = round(
            f1_score(dataframe["category"].to_numpy()[mask], predicted_categories[mask], average="macro"), 4
        )
    return language_scores


def main():
    parser = argparse.ArgumentParser(description="Score the full serving pipeline on validation data.")
    parser.add_argument("--validation-csv", required=True)
    arguments = parser.parse_args()
    dataframe = load_validation(arguments.validation_csv)
    predictor = TicketPredictor("evaluation")
    predictions = predictor.predict_tickets(build_tickets(dataframe))
    report = score_predictions(dataframe, predictions)
    report["category_f1_by_language"] = score_by_language(dataframe, predictions)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
