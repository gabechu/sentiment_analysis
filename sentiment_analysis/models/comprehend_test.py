from mock import Mock
from moto import mock_cognitoidentity
from pytest import fixture, raises

from .comprehend import Comprehend


@fixture
def fake_response():
    return {
        "Sentiment": "POSITIVE",
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


@mock_cognitoidentity
def test_comprehend_init():
    actual = Comprehend()
    assert actual.aws_region == "us-west-2"


@mock_cognitoidentity
def test_comprehend_detect_sentiment(fake_response):
    mock_predict = Mock()
    mock_predict.return_value = fake_response

    model = Comprehend()
    model._predict = mock_predict

    actual = model.detect_sentiment("positive", "en")
    fake_response.pop("ResponseMetadata")
    assert actual == fake_response

    mock_predict.assert_called_once_with(Text="positive", LanguageCode="en")


@mock_cognitoidentity
def test_comprehend_detect_sentiment_non_support_language_code():
    model = Comprehend()
    with raises(ValueError) as err:
        model.detect_sentiment("Goedemorgen", "nl")
    assert str(err.value) == (
        "Do not support language code nl, supports only "
        "['de', 'en', 'es', 'it', 'pt', 'fr', 'ja', 'ko', 'hi', 'ar', 'zh', 'zh-TW']."
    )
