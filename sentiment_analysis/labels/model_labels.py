from enum import Enum
from typing import TypedDict


class SentimentScore(TypedDict):
    Positive: float
    Negative: float
    Neutral: float
    Mixed: float


class ComprehendResults(TypedDict):
    # Sentiment can have these 4 labels:
    # POSITIVE, NEGATIVE, NEUTRAL and MIXED
    Sentiment: str
    SentimentScore: SentimentScore


class ComprehendLabel(Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    MIXED = "MIXED"
