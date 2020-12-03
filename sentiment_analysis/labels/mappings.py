from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from google.cloud.language_v1.types.language_service import AnalyzeSentimentResponse

from .dataset_labels import SemEvalSubTaskALabel, SpanishAirlinesTweetsLabel
from .model_labels import ComprehendLabel, ComprehendResults


class LabelMapper(ABC):
    @abstractmethod
    def map(model_results: Any) -> Enum:
        ...


class ComprehendLabelToSpanishAirlinesTweetsLabel(LabelMapper):
    def map(self, model_results: ComprehendResults,) -> SpanishAirlinesTweetsLabel:
        predicted_sentiment: str = model_results["Sentiment"].lower()

        if predicted_sentiment in SpanishAirlinesTweetsLabel.__members__:
            return SpanishAirlinesTweetsLabel(predicted_sentiment)
        return SpanishAirlinesTweetsLabel.unknown


class ComprehendLabelToSemEvalSubtaskALabel(LabelMapper):
    def map(self, model_results: ComprehendResults) -> SpanishAirlinesTweetsLabel:
        predicted_sentiment: str = model_results["Sentiment"]

        if predicted_sentiment in SemEvalSubTaskALabel.__members__:
            return SemEvalSubTaskALabel(predicted_sentiment)
        return ComprehendLabel.unknown


class GoogleNaturalLanguageLabelToSpanishAirlinesTweetsLabel(LabelMapper):
    def __init__(
        self,
        positive_neutral_cutoff: float = 0.25,
        negative_neutral_cutoff: float = -0.25,
    ):
        ...

    def map(
        self, model_results: AnalyzeSentimentResponse,
    ) -> SpanishAirlinesTweetsLabel:
        score = model_results.document_sentiment.score

        if score > self.positive_neutral_cutoff:
            label = "positive"
        elif score < self.negative_neutral_cutoff:
            label = "negative"
        else:
            label = "neutral"

        return SpanishAirlinesTweetsLabel(label)


class GoogleNaturalLanguageLabelToSemEvalSubtaskALabel(LabelMapper):
    def __init__(
        self,
        positive_neutral_cutoff: float = 0.25,
        negative_neutral_cutoff: float = -0.25,
    ):
        ...

    def map(self, model_results: AnalyzeSentimentResponse,) -> SemEvalSubTaskALabel:
        score = model_results.document_sentiment.score

        if score > self.positive_neutral_cutoff:
            label = "POSITIVE"
        elif score < self.negative_neutral_cutoff:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return SemEvalSubTaskALabel(label)
