from google.cloud.language_v1 import Sentence, Sentiment, TextSpan
from google.cloud.language_v1.types.language_service import AnalyzeSentimentResponse
from mock import patch
from pytest import fixture

from .google_natural_language import GoogleNaturalLanguage


@fixture
def fake_response() -> AnalyzeSentimentResponse:
    document_sentiment = Sentiment(score=1.600000023841858, magnitude=0.800000011920929)
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


def test_googlenaturallanguage_init():
    actual = GoogleNaturalLanguage()

    assert actual.supported_languages == [
        "ar",
        "zh",
        "zh-Hant",
        "nl",
        "en",
        "fr",
        "de",
        "id",
        "it",
        "ja",
        "ko",
        "pt",
        "es",
        "th",
        "tr",
        "vi",
    ]


@patch("sentiment_analysis.models.google_natural_language.GoogleNaturalLanguage.client")
def test_googlenaturallanguage_detect_sentiment(mock_client, fake_response):
    mock_client.analyze_sentiment.return_value = fake_response

    model = GoogleNaturalLanguage()
    actual = model.detect_sentiment(
        "Don't listen to what the critics have always said about this cute, charming "
        "little movie. Madonna is GREAT in this clever comedy."
    )
    assert actual == fake_response


def test_googlenaturallanguage_parse_response():
    ...
