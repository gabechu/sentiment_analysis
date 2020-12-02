from typing import TypedDict


class SentimentScore(TypedDict):
    Positive: float
    Negative: float
    Neutral: float
    Mixed: float


class ComprehendResults(TypedDict):
    Sentiment: str
    SentimentScore: SentimentScore
