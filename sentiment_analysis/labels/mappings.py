from google.cloud.language_v1.types.language_service import AnalyzeSentimentResponse

from .dataset_labels import SemEvalSubTaskALabel, SpanishArilinesTweetsLabel
from .model_labels import ComprehendResults, ComprehendLabel
from typing import Union


class ComprehendResultsMapper(object):
    def to_spanish_airlines_tweets_label(
        self, model_results: ComprehendResults,
    ) -> Union[SpanishArilinesTweetsLabel, ComprehendLabel]:
        # keys are model labels and values are dataset labels
        # no mapping for MIXED, which means any MIXED label would be count as wrong
        # classifications in evaluation.
        mapping = {"POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral"}
        predicted_sentiment: str = model_results["Sentiment"]

        if predicted_sentiment in mapping:
            return SpanishArilinesTweetsLabel(mapping[predicted_sentiment])
        else:
            return ComprehendLabel(predicted_sentiment)

    def to_semeval_subtask_a_label(
        self, model_results: ComprehendResults,
    ) -> Union[SemEvalSubTaskALabel, ComprehendLabel]:
        predicted_sentiment: str = model_results["Sentiment"]

        if predicted_sentiment in SemEvalSubTaskALabel.__members__:
            return SemEvalSubTaskALabel(predicted_sentiment)
        else:
            return ComprehendLabel(predicted_sentiment)


class GoogleNaturalLangaugeResultsMapper(object):
    positive_neutral_cutoff = 0.25
    negative_neutral_cutoff = -0.25

    def to_spanish_airlines_tweets(
        self, model_results: AnalyzeSentimentResponse,
    ) -> SpanishArilinesTweetsLabel:
        score = model_results.document_sentiment.score

        if score > self.positive_neutral_cutoff:
            str_label = "positive"
        elif score < self.negative_neutral_cutoff:
            str_label = "negative"
        else:
            str_label = "neutral"

        return SpanishArilinesTweetsLabel(str_label)

    def google_natural_language_results_to_semeval_subtask_a(
        self, model_results: AnalyzeSentimentResponse,
    ) -> SemEvalSubTaskALabel:
        score = model_results.document_sentiment.score

        if score > self.positive_neutral_cutoff:
            str_label = "POSITIVE"
        elif score < self.negative_neutral_cutoff:
            str_label = "NEGATIVE"
        else:
            str_label = "NEUTRAL"

        return SemEvalSubTaskALabel(str_label)
