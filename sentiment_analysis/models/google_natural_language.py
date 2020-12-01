from typing import Any, Dict, List

from decouple import config
from google.cloud import language_v1
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
    def client(self):
        return language_v1.LanguageServiceClient.from_service_account_json(
            self.__credentials_path
        )

    def config_request(self, text: str, language_code: str) -> Dict:
        return {
            "document": {
                "content": text,
                "type_": language_v1.Document.Type.PLAIN_TEXT,
                "language": language_code,
            },
            "encoding_type": language_v1.EncodingType.UTF8,
        }

    def parse_protobuf(self, response: AnalyzeSentimentResponse) -> Dict:
        """
        The structure of the returned dict:
            document_sentiment
                - score
                - magnitude
            sentences
                - text
                    - content
                    - begin_offset
                - sentiment
                    - score
                    - magnitude
        """
        result: Dict[str, Any] = {
            "document_sentiment": {
                "score": response.document_sentiment.score,
                "magnitude": response.document_sentiment.magnitude,
            }
        }

        sentences_predictions: List[Dict] = []
        for sentence in response.sentences:
            prediction: Dict = {
                "text": {
                    "content": sentence.text.content,
                    "begin_offset": sentence.text.begin_offset,
                },
                "sentiment": {
                    "score": sentence.sentiment.score,
                    "magnitude": sentence.sentiment.magnitude,
                },
            }
            sentences_predictions.append(prediction)
        result["sentences"] = sentences_predictions

        return result

    def detect_sentiment(self, text: str, language_code: str = "en") -> Dict:
        request = self.config_request(text, language_code)
        return self.client.analyze_sentiment(request=request)
