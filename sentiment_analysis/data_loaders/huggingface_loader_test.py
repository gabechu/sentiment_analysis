from pytest import raises, fixture
from mock import patch

from .huggingface_loader import HuggingfaceLoader
from datasets import DatasetDict
from datasets.arrow_dataset import Dataset
from pyarrow.lib import Table
import pandas as pd


@fixture
def dataset():
    arrow_table = Table.from_pydict({"label": [1, 0], "text": ["good", "bad"]})
    return Dataset(arrow_table)


@fixture
def dataset_dict(dataset):
    return DatasetDict({"test": dataset})


def test_huggingfaceloader_init():
    actual = HuggingfaceLoader("imdb")
    assert actual.dataset_name == "imdb"


def test_huggingfaceloader_init_with_unsupported_dataset():
    with raises(ValueError) as err:
        HuggingfaceLoader("NotSupported")

    assert str(err.value) == (
        "Dataset NotSupported not found, supports only "
        "['allocine', 'imdb', 'rotten_tomatoes', 'sentiment140', 'yelp_polarity']."
    )


@patch("sentiment_analysis.data_loaders.huggingface_loader.load_dataset")
def test_huggingfaceloader_load_test_data_with_datasetdict(
    mock_load_dataset, dataset_dict
):
    mock_load_dataset.return_value = dataset_dict
    loader = HuggingfaceLoader("imdb")
    actual = loader.load_test_data()

    assert actual.equals(
        pd.DataFrame.from_dict({"label": [1, 0], "text": ["good", "bad"]})
    )


@patch("sentiment_analysis.data_loaders.huggingface_loader.load_dataset")
def test_huggingfaceloader_load_test_data_without_testset(mock_load_dataset, dataset):
    mock_load_dataset.return_value = dataset
    loader = HuggingfaceLoader("imdb")
    with raises(KeyError) as err:
        loader.load_test_data()

    # KeyError adds extra quotes to the error message
    # https://stackoverflow.com/questions/24998968/why-does-strkeyerror-add-extra-quotes
    assert str(err.value) == "'imdb does not have a test set.'"
