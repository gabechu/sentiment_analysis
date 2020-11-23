from .huggingface_loader import HuggingfaceLoader


class RottenTomatoesLoader(HuggingfaceLoader):
    def __init__(self):
        super().__init__("sentiment140")
