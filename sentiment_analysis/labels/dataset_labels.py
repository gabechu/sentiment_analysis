from enum import Enum


class SpanishAirlinesTweetsLabel(Enum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"
    other = "other"


class SemEvalSubTaskALabel(Enum):
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
    OTHER = "OTHER"
