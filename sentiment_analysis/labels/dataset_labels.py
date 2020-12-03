from enum import Enum


class SpanishAirlinesTweetsLabel(Enum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"
    unknown = "unknown"


class SemEvalSubTaskALabel(Enum):
    NEGATIVE = "NEGATIVE"
    NETRUAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
    UNKNOWN = "UNKNOWN"
