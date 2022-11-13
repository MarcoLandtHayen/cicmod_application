import numpy as np
import pytest

from numpy.testing import assert_almost_equal

from cicmod_application.events import consecutive_masking, el_nino, la_nina


# define dtype check as suggested by `abarnert` https://stackoverflow.com/questions/26921836/correct-way-to-test-for-numpy-dtype
dtype_test = (
    lambda xx, dtype: True
    if isinstance(xx, np.ndarray) and xx.dtype == dtype and xx.flags.contiguous
    else False
)


@pytest.fixture
def example_ENSO_ndarray():

    data = np.array([1, 1, 1, 0, -1, -2, -3, -0.5, 2, 1, 0, 0.5, 2, 2, 2, 2, 2])

    return data


@pytest.fixture
def example_el_nino_ndarray():

    data = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    return data


@pytest.fixture
def example_la_nina_ndarray():

    data = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    return data


example_consecutive = np.array(
    [
        [[1, 1, 1, 1], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1]],
    ]
)

result_consecutive_axis0 = np.array(
    [
        [[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    ]
)

result_consecutive_axis1 = np.array(
    [
        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1]],
    ]
)

result_consecutive_axis2 = np.array(
    [
        [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ]
)


def test_consecutive_event_default(mask=example_consecutive):
    """Ensures that the consective_event works in default setting."""
    actual = consecutive_masking(mask=mask)

    # check if values are almost equal
    assert_almost_equal(
        actual=actual,
        desired=mask,
        decimal=3,
    )
    # check for dtype
    print(actual.astype(bool))
    assert dtype_test(actual, bool)


@pytest.mark.parametrize(
    "axis, mel, should",
    [
        (0, 2, result_consecutive_axis0),
        (1, 2, result_consecutive_axis1),
        (2, 2, result_consecutive_axis2),
    ],
)
def test_consecutive_event_axis(axis, should, mel, mask=example_consecutive):
    """Ensures that the consective_event works in default setting."""
    actual = consecutive_masking(mask=mask, axis=axis, min_event_length=mel)
    print(actual)
    # check if values are almost equal
    assert_almost_equal(
        actual=actual,
        desired=should,
        decimal=3,
    )
    # check for dtype
    assert dtype_test(actual, bool)
