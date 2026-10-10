import sys
from collections import Counter

import pandas as pd

from app.predictor import TicketPredictor


def load_validation(csv_path):
    dataframe = pd.read_csv(csv_path, encoding="utf-8", dtype=str, keep_default_na=False)
    dataframe["secondary_category"] = dataframe["secondary_category"].str.strip().replace({"": "none"})
    return dataframe


def build_tickets(dataframe):
    return [
        {"channel": row["channel"], "subject": row["subject"], "text": row["text"]}
        for _, row in dataframe.iterrows()
    ]


def count_raw_secondaries(predictor, tickets):
    raw_inputs = predictor.build_classic_inputs(tickets, predictor.text_choice["secondary_category"])
    return Counter(str(label) for label in predictor.secondary_model.predict(raw_inputs))


def count_final_secondaries(predictor, tickets):
    predictions = predictor.predict_tickets(tickets)
    return Counter(prediction["secondary_category"] or "none" for prediction in predictions)


validation_data = load_validation(sys.argv[1])
tickets = build_tickets(validation_data)
predictor = TicketPredictor("check")
print("true secondary labels:  ", dict(Counter(validation_data["secondary_category"])))
print("raw model predictions:  ", dict(count_raw_secondaries(predictor, tickets)))
print("after consistency rules:", dict(count_final_secondaries(predictor, tickets)))