from typing import Dict


def comprehend_to_checklist_predictions(comprehend_prediction: Dict):
    mapping = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2, "MIXED": 3}
    int_label = mapping[comprehend_prediction["Sentiment"]]
    scores = comprehend_prediction["SentimentScore"]
    confs = [
        scores["Negative"],
        scores["Neutral"],
        scores["Positive"]
    ]

    rescaled_confs = [round(v/sum(confs), 6) for v in confs]

    return f"{int_label} {rescaled_confs[0]} {rescaled_confs[1]} {rescaled_confs[2]}"
