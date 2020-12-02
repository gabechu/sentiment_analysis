from typing import Dict, Callable, List

from decouple import config
from google.cloud import language_v1
from google.cloud.language_v1 import LanguageServiceClient
from google.cloud.language_v1.types.language_service import AnalyzeSentimentResponse


class GoogleNaturalLanguage(object):
    __credentials_path = config("GOOGLE_APPLICATION_CREDENTIALS")

    def __init__(self):
        # https://cloud.google.com/natural-language/docs/languages#sentiment_analysis
        self.supported_languages = [
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

    @property
    def client(self) -> LanguageServiceClient:
        return language_v1.LanguageServiceClient.from_service_account_json(
            self.__credentials_path
        )

    def _config_request(self, text: str, language_code: str) -> Dict:
        return {
            "document": {
                "content": text,
                "type_": language_v1.Document.Type.PLAIN_TEXT,
                "language": language_code,
            },
            "encoding_type": language_v1.EncodingType.UTF8,
        }

    def detect_sentiment(
        self, text: str, language_code: str = "en"
    ) -> AnalyzeSentimentResponse:
        if language_code not in self.supported_languages:
            raise ValueError(
                f"Language code {language_code} is not supported by Google NL."
            )
        request = self._config_request(text, language_code)
        return self.client.analyze_sentiment(request=request)
