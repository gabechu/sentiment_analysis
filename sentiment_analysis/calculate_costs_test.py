from numpy.testing import assert_almost_equal

from .calculate_costs import calculate_comprehend_cost_no_batch


def test_calculate_comprehend_cost_no_batch_single_text():
    text = ["test"]
    actual = calculate_comprehend_cost_no_batch(text)
    assert_almost_equal(actual, 0.0003)


def test_calculate_comprehend_cost_no_batch_less_than_3_unit_per_request():
    texts = ["test", "test", "test"]
    actual = calculate_comprehend_cost_no_batch(texts)
    assert_almost_equal(actual, 0.0009)


def test_calculate_comprehend_cost_no_batch_equal_to_3_unit_per_request():
    texts = ["t" * 300]
    actual = calculate_comprehend_cost_no_batch(texts)
    assert_almost_equal(actual, 0.0003)


def test_calculate_comprehend_cost_no_batch_greater_than_3_unit_per_request():
    texts = ["t" * 550] * 10000
    actual = calculate_comprehend_cost_no_batch(texts)
    assert_almost_equal(actual, 6.0)
