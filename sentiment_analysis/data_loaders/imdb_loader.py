from .huggingface_loader import HuggingfaceLoader


class ImdaLoader(HuggingfaceLoader):
    def __init__(self):
        super().__init__("imdb")
