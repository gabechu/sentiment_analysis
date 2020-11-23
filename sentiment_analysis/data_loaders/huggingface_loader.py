from typing import Optional, Union

from datasets import DatasetDict, load_dataset
from datasets.arrow_dataset import Dataset
from pandas import DataFrame


class HuggingfaceLoader(object):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data: Optional[Union[DatasetDict, Dataset]] = None

    def load_data(self) -> Union[DatasetDict, Dataset]:
        return load_dataset(self.dataset_name)

    def load_test_data(self) -> DataFrame:
        if not self.data:
            self.data = self.load_data()

        test_data = self.data["test"].data
        return test_data.to_pandas()
