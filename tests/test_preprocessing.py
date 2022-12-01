import numpy as np
import pytest

from cicmod_application.preprocessing import split_sequence


@pytest.fixture
def example_ndarray():

    data = np.array(
        [
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 8],
        ]
    )

    return data


@pytest.fixture
def example_split_ndarray():

    data = np.array([[[1, 5], [2, 6], [3, 7]], [[2, 6], [3, 7], [4, 8]]])

    return data


def test_split_sequence(example_ndarray, example_split_ndarray):
    """Ensure that function split_sequence splits given sequence correctly into samples of specified length."""

    assert (split_sequence(example_ndarray, 3) == example_split_ndarray).all()
