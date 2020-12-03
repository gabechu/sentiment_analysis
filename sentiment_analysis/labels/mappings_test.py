import pytest
from google.cloud.language_v1 import Sentence, Sentiment, TextSpan
from google.cloud.language_v1.types.language_service import AnalyzeSentimentResponse
from sentiment_analysis.labels.model_labels import ComprehendResults

from .mappings import (
    ComprehendLabelToSemEvalSubtaskALabel,
    ComprehendLabelToSpanishAirlinesTweetsLabel,
    GoogleNaturalLanguageLabelToSpanishAirlinesTweetsLabel,
    GoogleNaturalLanguageLabelToSemEvalSubtaskALabel,
)


# Test data preparation
def get_comprehend_results(sentiment: str) -> ComprehendResults:
    # choices of sentiment are POSITIVE, NEGATIVE and NEUTRAL
    return {
        "Sentiment": sentiment,
        "SentimentScore": {
            "Positive": 0.8893707394599915,
            "Negative": 0.00359430443495512,
            "Neutral": 0.037097539752721786,
            "Mixed": 0.06993737816810608,
        },
        "ResponseMetadata": {
            "RequestId": "72023c95-95bb-424a-9ee0-17325dfc852a",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
                "x-amzn-requestid": "72023c95-95bb-424a-9ee0-17325dfc852a",
                "content-type": "application/x-amz-json-1.1",
                "content-length": "163",
                "date": "Thu, 26 Nov 2020 22:56:36 GMT",
            },
            "RetryAttempts": 0,
        },
    }


def get_analyze_sentiment_response(doc_score: float) -> AnalyzeSentimentResponse:
    document_sentiment = Sentiment(score=doc_score, magnitude=0.800000011920929)
    sentence_1 = Sentence(
        text=TextSpan(
            content=(
                "Don't listen to what the critics have always said about this cute, "
                "charming little movie."
            ),
            begin_offset=0,
        ),
        sentiment=Sentiment(score=0.6000000238418579, magnitude=0.6000000238418579),
    )
    sentence_2 = Sentence(
        text=TextSpan(
            content="Madonna is GREAT in this clever comedy.", begin_offset=90,
        ),
        sentiment=Sentiment(score=0.8999999761581421, magnitude=0.8999999761581421),
    )

    return AnalyzeSentimentResponse(
        document_sentiment=document_sentiment, sentences=[sentence_1, sentence_2]
    )


# Tests start here
@pytest.mark.parametrize(
    "test_label, expected",
    [
        ("POSITIVE", "positive"),
        ("NEUTRAL", "neutral"),
        ("NEGATIVE", "negative"),
        ("MIXED", "other"),
    ],
)
def test_comprehend_labe_to_spanish_airlines_tweets_label(test_label, expected):
    comprehend_results = get_comprehend_results(test_label)
    mapper = ComprehendLabelToSpanishAirlinesTweetsLabel()

    actual = mapper.map(comprehend_results)
    assert actual.value == expected


@pytest.mark.parametrize(
    "test_label, expected",
    [
        ("POSITIVE", "POSITIVE"),
        ("NEUTRAL", "NEUTRAL"),
        ("NEGATIVE", "NEGATIVE"),
        ("MIXED", "OTHER"),
    ],
)
def test_comprehend_label_to_semeval_subtask_a_label(test_label, expected):
    comprehend_results = get_comprehend_results(test_label)
    mapper = ComprehendLabelToSemEvalSubtaskALabel()

    actual = mapper.map(comprehend_results)
    assert actual.value == expected


@pytest.mark.parametrize(
    "test_score, expected",
    [
        (0.8, "positive"),
        (0.0, "neutral"),
        (-0.8, "negative"),
        (0.25, "neutral"),
        (-0.25, "neutral"),
    ],
)
def test_google_natural_language_label_to_spanish_airlines_tweets_label(
    test_score, expected
):
    google_results = get_analyze_sentiment_response(test_score)
    mapper = GoogleNaturalLanguageLabelToSpanishAirlinesTweetsLabel()

    actual = mapper.map(google_results)
    assert actual.value == expected


@pytest.mark.parametrize(
    "test_score, expected",
    [
        (0.8, "POSITIVE"),
        (0.0, "NEUTRAL"),
        (-0.8, "NEGATIVE"),
        (0.25, "NEUTRAL"),
        (-0.25, "NEUTRAL"),
    ],
)
def test_google_natural_language_label_to_semeval_subtask_a_label(test_score, expected):
    google_results = get_analyze_sentiment_response(test_score)
    mapper = GoogleNaturalLanguageLabelToSemEvalSubtaskALabel()

    actual = mapper.map(google_results)
    assert actual.value == expected
