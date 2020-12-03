import pytest
from google.cloud.language_v1 import Sentence, Sentiment, TextSpan
from google.cloud.language_v1.types.language_service import AnalyzeSentimentResponse
from sentiment_analysis.labels.model_labels import ComprehendResults

from .mappings import ComprehendResultsMapper, GoogleNaturalLangaugeResultsMapper


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


@pytest.mark.parametrize(
    "test_label, expected",
    [
        ("POSITIVE", "positive"),
        ("NEUTRAL", "neutral"),
        ("NEGATIVE", "negative"),
        ("MIXED", "MIXED"),
    ],
)
def test_comprehendresultsmapper_to_spanish_airlines_tweets_label(test_label, expected):
    comprehend_results = get_comprehend_results(test_label)
    mapper = ComprehendResultsMapper()

    actual = mapper.to_spanish_airlines_tweets_label(comprehend_results)
    assert actual.value == expected


def test_comprehendresultsmapper_to_spanish_airlines_tweets_label_invalid_sentiment():
    comprehend_results = get_comprehend_results("InvalidLabel")
    mapper = ComprehendResultsMapper()

    with pytest.raises(ValueError) as err:
        mapper.to_spanish_airlines_tweets_label(comprehend_results)
    assert str(err.value) == "'InvalidLabel' is not a valid ComprehendLabel"


@pytest.mark.parametrize(
    "test_label, expected",
    [
        ("POSITIVE", "POSITIVE"),
        ("NEUTRAL", "NEUTRAL"),
        ("NEGATIVE", "NEGATIVE"),
        ("MIXED", "MIXED"),
    ],
)
def test_comprehendresultsmapper_to_semeval_subtask_a_label_positive(
    test_label, expected
):
    comprehend_results = get_comprehend_results(test_label)
    mapper = ComprehendResultsMapper()

    actual = mapper.to_semeval_subtask_a_label(comprehend_results)
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
def test_googlenaturallangaugeresultsmapper_to_spanish_airlines_tweets_label(
    test_score, expected
):
    google_results = get_analyze_sentiment_response(test_score)
    mapper = GoogleNaturalLangaugeResultsMapper()

    actual = mapper.to_spanish_airlines_tweets(google_results)
    assert actual.value == expected
