from typing import Union

from datasets import DatasetDict, load_dataset
from datasets.arrow_dataset import Dataset
from pandas import DataFrame
from pyarrow.lib import Table


class HuggingfaceLoader(object):
    # we are only interested in "label" and "text" for sentiment analysis
    filter_mapping = {"sentiment140": ["text", "sentiment"]}

    # Rename corresponding columns to literal label and text if they are not
    rename_mapping = {
        "allocine": {"review": "text"},
        "sentiment140": {"sentiment": "label"},
    }

    def __init__(self, dataset_name: str):
        supported_datasets = [
            "allocine",
            "imdb",
            "rotten_tomatoes",
            "sentiment140",
            "yelp_polarity",
        ]
        if dataset_name in supported_datasets:
            self.dataset_name = dataset_name
        else:
            raise ValueError(
                f"Dataset {dataset_name} not found, "
                f"supports only {supported_datasets}."
            )

    def load_data(self):
        self.data = load_dataset(self.dataset_name)

    def load_test_data(self) -> DataFrame:
        data: Union[DatasetDict, Dataset] = load_dataset(self.dataset_name)

        if "test" not in data:
            raise KeyError(f"{self.dataset_name} does not have a test set.")

        test_data: Table = data["test"].data
        df = test_data.to_pandas()

        # filter columns
        if self.dataset_name in self.filter_mapping:
            df = df.filter(items=self.filter_mapping[self.dataset_name])

        # rename columns
        if self.dataset_name in self.rename_mapping:
            df = df.rename(columns=self.rename_mapping[self.dataset_name])

        # avoid randomness on column orders
        return df.sort_index(axis=1)
