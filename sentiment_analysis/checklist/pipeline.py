from typing import Dict, List

from sentiment_analysis.models.comprehend import Comprehend
from tqdm import tqdm

from .convert_predictions import comprehend_to_checklist_predictions


def get_model() -> Comprehend:
    return Comprehend()


def get_clean_text(text: str) -> str:
    return text.rstrip()


def load_test_file(file_path: str) -> List[str]:
    with open(file_path) as f:
        raw_texts = f.readlines()
        clean_texts = list(map(lambda x: get_clean_text(x), raw_texts))
    return clean_texts


def make_prediction(text: str, model: Comprehend, language_code: str = "en"):
    return model.detect_sentiment(text, language_code)


def dump_predictions_to_file(file_path: str, predictions: Dict):
    with open(file_path, "w") as f:
        for pred in predictions:
            str_pred = comprehend_to_checklist_predictions(pred)
            f.write(f"{str_pred}\n")


def dump_predictions_pipeline():
    test_text_file = "/Users/gabechu/Code/zendesk/sentiment_analysis/release_data/sentiment/tests_n500"
    dump_preds_file = "/Users/gabechu/Code/zendesk/sentiment_analysis/comprehend"

    model = get_model()
    texts = load_test_file(test_text_file)

    preds = [make_prediction(text, model) for text in tqdm(texts)]

    dump_predictions_to_file(dump_preds_file, preds)


if __name__ == "__main__":
    dump_predictions_pipeline()
