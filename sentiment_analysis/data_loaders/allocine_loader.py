from .huggingface_loader import HuggingfaceLoader


class AllocineLoader(HuggingfaceLoader):
    def __init__(self):
        super().__init__("allocine")
