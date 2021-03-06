from typing import Dict, List

import pandas as pd
from datasets import DatasetDict
from datasets.arrow_dataset import Dataset
from mock import patch
from pyarrow.lib import Table
from pytest import raises

from .huggingface_loader import HuggingfaceLoader


def construct_dataset(data: Dict[str, List]):
    arrow_table = Table.from_pydict(data)
    return Dataset(arrow_table)


def construct_dataset_dict(data: Dict[str, List]):
    dataset = construct_dataset(data)
    return DatasetDict({"test": dataset})


def test_huggingfaceloader_init():
    actual = HuggingfaceLoader("imdb")
    assert actual.dataset_name == "imdb"


@patch("sentiment_analysis.data_loaders.huggingface_loader.load_dataset")
def test_huggingfaceloader_load_test_data_with_datasetdict(mock_load_dataset):
    mock_load_dataset.return_value = construct_dataset_dict(
        {"label": [1, 0], "text": ["good", "bad"]}
    )

    loader = HuggingfaceLoader("imdb")
    actual = loader.load_test_data()
    expect = pd.DataFrame.from_dict({"label": [1, 0], "text": ["good", "bad"]})

    assert actual.equals(expect.sort_index(axis=1))


@patch("sentiment_analysis.data_loaders.huggingface_loader.load_dataset")
def test_huggingfaceloader_load_test_data_without_testset(mock_load_dataset):
    mock_load_dataset.return_value = construct_dataset(
        {"label": [1, 0], "text": ["good", "bad"]}
    )

    loader = HuggingfaceLoader("imdb")
    with raises(KeyError) as err:
        loader.load_test_data()

    # KeyError adds extra quotes to the error message
    # https://stackoverflow.com/questions/24998968/why-does-strkeyerror-add-extra-quotes
    assert str(err.value) == "'imdb does not have a test set.'"
