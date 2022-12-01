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

    data = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=bool)

    return data


@pytest.fixture
def example_la_nina_ndarray():

    data = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)

    return data


@pytest.fixture
def example_consecutive():
    data = np.array(
        [
            [[1, 1, 1, 1], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1]],
        ],
        dtype=bool,
    )
    return data


@pytest.fixture
def result_consecutive_axis0():
    data = np.array(
        [
            [[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=bool,
    )
    return data


@pytest.fixture
def result_consecutive_axis1():
    data = np.array(
        [
            [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1]],
        ],
        dtype=bool,
    )
    return data


@pytest.fixture
def result_consecutive_axis2():
    data = np.array(
        [
            [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=bool,
    )
    return data


def test_consecutive_masking_default(example_consecutive):
    """Ensures that the consective_event works in default setting."""
    actual = consecutive_masking(mask=example_consecutive)

    # check if values are almost equal
    assert_almost_equal(
        actual=actual,
        desired=example_consecutive,
        decimal=3,
    )
    # check for dtype
    assert dtype_test(actual, bool)


@pytest.mark.parametrize(
    "axis",
    [0, 1, 2],
)
def test_consecutive_masking_axis(
    axis,
    example_consecutive,
    result_consecutive_axis0,
    result_consecutive_axis1,
    result_consecutive_axis2,
):
    """Ensures that the consective_event works in along different axis."""
    # Set minimal event length
    mel = 2

    if axis == 0:
        should = result_consecutive_axis0
    elif axis == 1:
        should = result_consecutive_axis1
    elif axis == 2:
        should = result_consecutive_axis2

    actual = consecutive_masking(
        mask=example_consecutive, axis=axis, min_event_length=mel
    )
    # check if values are almost equal
    assert_almost_equal(
        actual=actual,
        desired=should,
        decimal=3,
    )
    # check for dtype
    assert dtype_test(actual, bool)


def test_consecutive_masking_int(example_consecutive):
    """Ensures that the consective_event can return int."""
    axis = 2
    mel = 2
    dtype = int
    should = np.array(
        [
            [[1, 2, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )

    actual = consecutive_masking(
        mask=example_consecutive,
        min_event_length=mel,
        axis=axis,
        dtype=dtype,
    )

    # check if values are almost equal if they are
    assert_almost_equal(
        actual=actual,
        desired=should,
        decimal=3,
    )
    # check for dtype
    assert dtype_test(actual, dtype)


def test_consecutive_masking_float(example_consecutive):
    """Ensures that the consective_event can return int."""
    axis = 2
    mel = 2
    dtype = float

    mask = example_consecutive.astype(dtype)
    should = np.array(
        [
            [[1, 2, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=float,
    )

    actual = consecutive_masking(
        mask=mask,
        min_event_length=mel,
        axis=axis,
        dtype=dtype,
    )

    # check if values are almost equal if they are
    assert_almost_equal(
        actual=actual,
        desired=should,
        decimal=3,
    )
    # check for dtype
    assert dtype_test(actual, dtype)


def test_consecutive_masking_fails(example_consecutive):
    """Ensures that the consective_event raises an error if minimum event length is too large."""
    axis = 2
    mel = example_consecutive.shape[axis] + 1
    with pytest.raises(Exception) as e_info:
        consecutive_masking(mask=example_consecutive, min_event_length=mel, axis=axis)


def test_el_nino(example_ENSO_ndarray, example_el_nino_ndarray):
    """Ensures that el_nino can select el_nino properly."""
    actual = el_nino(example_ENSO_ndarray, min_event_length=3, threshold=1)
    assert_almost_equal(
        actual=actual,
        desired=example_el_nino_ndarray,
        decimal=3,
    )
    # check for dtype
    assert dtype_test(actual, bool)


def test_la_nina(example_ENSO_ndarray, example_la_nina_ndarray):
    """Ensures that el_nino can select la_nina properly."""
    actual = la_nina(example_ENSO_ndarray, min_event_length=3, threshold=-1)
    assert_almost_equal(
        actual=actual,
        desired=example_la_nina_ndarray,
        decimal=3,
    )
    # check for dtype
    assert dtype_test(actual, bool)
