from typing import Optional, Union

from datasets import DatasetDict, load_dataset
from datasets.arrow_dataset import Dataset
from pandas import DataFrame


# TODO: add tests
class HuggingfaceLoader(object):
    filter_mapping = {
        "sentiment140": ["text", "sentiment"]
    }

    rename_mapping = {
        "allocine": {"review": "text"},
        "sentiment140": {"sentiment": "label"}
    }

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data: Optional[Union[DatasetDict, Dataset]] = None

    def _rename_columns(self, df: DataFrame) -> DataFrame:
        if self.dataset_name in self.rename_mapping:
            return df.rename(columns=self.rename_mapping[self.dataset_name])
        else:
            return df

    def _filter_columns(self, df: DataFrame) -> DataFrame:
        if self.dataset_name in self.filter_mapping:
            return df.filter(items=self.filter_mapping[self.dataset_name])
        else:
            return df

    def load_data(self) -> Union[DatasetDict, Dataset]:
        return load_dataset(self.dataset_name)

    def load_test_data(self) -> DataFrame:
        if not self.data:
            self.data = self.load_data()

        test_data = self.data["test"].data
        df = test_data.to_pandas()

        df = self._filter_columns(df)
        df = self._rename_columns(df)

        return df
