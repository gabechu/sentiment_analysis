from .datasets_labels import SpanishArilinesTweetsLabel, SemEvalSubTaskALabel


def comprehend_results_to_spanish_airlines_tweets(model_results) -> SpanishAirlinesTweetsLabel:
    ...


def comprehend_results_to_semeval_subtask_a(model_results) -> SemEvalSubTaskALabel:
    ...


def google_natural_language_results_to_spanish_airlines_tweets(model_results) -> SpanishAirlinesTweetsLabel:
    ...


def google_natural_language_results_to_semeval_subtask_a(model_results) -> SemEvalSubTaskALabel:
    ...
