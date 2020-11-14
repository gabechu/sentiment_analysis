from typing import Dict

from decouple import config
from sentiment_analysis.aws.config_session import config_cognito_session


class Comprehend(object):
    IDENTITY_POOL_ID = config("IDENTITY_POOL_ID")
    AWS_REGION = "us-west-2"

    def __init__(self):
        comprehend = self._initiate_comprehend()
        self.model_func = comprehend.detect_sentiment

    def _initiate_comprehend(self):
        sess = config_cognito_session(self.IDENTITY_POOL_ID, self.AWS_REGION)
        return sess.client(service_name="comprehend", region_name=self.AWS_REGION)

    def detect_sentiment(self, text: str, language_code: str) -> Dict:
        response = self.model_func(Text=text, LanguageCode=language_code)

        keep_keys = {"Sentiment", "SentimentScore"}
        return {key: value for key, value in response.items() if key in keep_keys}
