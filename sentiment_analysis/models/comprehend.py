from typing import Dict

from botocore.client import BaseClient
from decouple import config
from sentiment_analysis.aws.config_session import config_cognito_session


class Comprehend(object):
    # read from .env in root
    __IDENTITY_POOL_ID = config("IDENTITY_POOL_ID")

    def __init__(self, aws_region: str = "us-west-2"):
        self.aws_region = aws_region
        self.supported_languages = [
            "de",
            "en",
            "es",
            "it",
            "pt",
            "fr",
            "ja",
            "ko",
            "hi",
            "ar",
            "zh",
            "zh-TW",
        ]

        comprehend = self._initiate_comprehend()
        self._predict = comprehend.detect_sentiment

    def _initiate_comprehend(self) -> BaseClient:
        sess = config_cognito_session(self.__IDENTITY_POOL_ID, self.aws_region)
        return sess.client(service_name="comprehend", region_name=self.aws_region)

    def detect_sentiment(self, text: str, language_code: str) -> Dict:
        if language_code not in self.supported_languages:
            raise ValueError(
                f"Do not support language code {language_code}, "
                f"supports only {self.supported_languages}."
            )

        # return with keys of ['Sentiment', 'SentimentScore', 'ResponseMetadata']
        response: Dict = self._predict(Text=text, LanguageCode=language_code)
        keep_keys = {"Sentiment", "SentimentScore"}

        return {key: value for key, value in response.items() if key in keep_keys}
