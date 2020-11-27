from typing import Union

from datasets import DatasetDict, load_dataset
from datasets.arrow_dataset import Dataset
from pandas import DataFrame
from pyarrow.lib import Table


class HuggingfaceLoader(object):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def load_test_data(self) -> DataFrame:
        data: Union[DatasetDict, Dataset] = load_dataset(self.dataset_name)

        if "test" not in data:
            raise KeyError(f"{self.dataset_name} does not have a test set.")

        test_data: Table = data["test"].data
        return test_data.to_pandas()
