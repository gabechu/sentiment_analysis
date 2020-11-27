from mock import mock_open, patch
from pytest import raises
from pandas import DataFrame

from .semeval_loader import SemEvalLoader


def test_semevalloader_init():
    actual = SemEvalLoader("test_file", "en", "A")

    assert actual.input_file == "test_file"
    assert actual.language == "en"
    assert actual.task_name == "A"


def test_semevalloader_init_invalid_taskname():
    with raises(ValueError) as err:
        SemEvalLoader("test_file", "en", "NotExist")
    assert str(err.value) == (
        "Do not support subtask NotExist, available subtasks are ['A', 'B', 'C']"
    )


def test_semevalloader_load_test_data_subtask_a():
    data = (
        "619969366986235905\tneutral\tOrder Go Set a Watchman in store or through our "
        "website before Tuesday and get it half price! #GSAW @GSAWatchmanBook "
        "https://t.co/KET6EGD1an\t\n619987808317407232\tpositive\tA portion of book "
        "sales from our Harper Lee/Go Set a Watchman release party on Mon. 7/13 will "
        "support @CAP_Tulsa and the great work they do.\t"
    )

    loader = SemEvalLoader("test_file", "en", "A")

    with patch("builtins.open", mock_open(read_data=data)):
        actual = loader.load_test_data()

    assert actual.equals(
        DataFrame.from_dict(
            {
                "id": ["619969366986235905", "619987808317407232"],
                "label": ["neutral", "positive"],
                "tweet": [
                    (
                        "Order Go Set a Watchman in store or through our website before"
                        " Tuesday and get it half price! #GSAW @GSAWatchmanBook "
                        "https://t.co/KET6EGD1an"
                    ),
                    (
                        "A portion of book sales from our Harper Lee/Go Set a Watchman "
                        "release party on Mon. 7/13 will support @CAP_Tulsa and the "
                        "great work they do."
                    ),
                ],
            }
        )
    )
