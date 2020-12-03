from enum import Enum
from typing import TypedDict


class SentimentScore(TypedDict):
    Positive: float
    Negative: float
    Neutral: float
    Mixed: float


HTTPHeaders = TypedDict(
    "HTTPHeaders",
    {"x-amzn-requestid": str, "content-type": str, "content-length": str, "date": str},
)


class ResponseMetadata(TypedDict):
    RequestId: str
    HTTPStatusCode: int
    HTTPHeaders: HTTPHeaders
    RetryAttempts: int


class ComprehendResults(TypedDict):
    # Sentiment can have these 4 labels:
    # POSITIVE, NEGATIVE, NEUTRAL and MIXED
    Sentiment: str
    SentimentScore: SentimentScore
    ResponseMetadata: ResponseMetadata


# Referece of comprehend labels
class ComprehendLabel(Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    MIXED = "MIXED"
