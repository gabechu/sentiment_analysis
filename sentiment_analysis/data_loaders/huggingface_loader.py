from typing import Union

from datasets import DatasetDict, load_dataset
from datasets.arrow_dataset import Dataset
from pandas import DataFrame
from pyarrow import Table


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
            raise Exception(
                f"Do not support dataset {dataset_name}, "
                f"only supports {supported_datasets}"
            )

    def load_data(self):
        self.data = load_dataset(self.dataset_name)

    def load_test_data(self) -> DataFrame:
        data: Union[DatasetDict, Dataset] = load_dataset(self.dataset_name)
        test_data: Table = data["test"].data
        df = test_data.to_pandas()

        # filter columns
        if self.dataset_name in self.filter_mapping:
            return df.filter(items=self.filter_mapping[self.dataset_name])

        # rename columns
        if self.dataset_name in self.rename_mapping:
            return df.rename(columns=self.rename_mapping[self.dataset_name])

        return df
