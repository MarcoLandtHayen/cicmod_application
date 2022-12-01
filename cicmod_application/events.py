import numpy as np


def consecutive_masking(mask, min_event_length=1, axis=0, dtype=bool):
    """
     Returns a result for which only elements of a consecutive event in the input maskare marked as True.
     The consecutive event needs to be at least min_event_length long.

    Parameters
    ----------
    mask : numpy.ndarray
        Input array mask.
        A dtype of bool is recommended.
    min_event_len : int
        Minimal event length. (Greater equal)
    axis : int, optional
        Axis along which the consecutivity of input values is computed.
        If `axis` is not specified, it defaults is 0.
    dtype : dtype, optional
        Type of the returned array.
        If `dtype` is not specified, it defaults is `bool` dtype.

    Returns
    -------
    numpy.ndarray.
        Array with True for all indices part of a consecutive event specified by arguments.

    Example
    --------
    (For simplicity True=1, False=0.
    >>> mask = np.array(
        [[[1, 1, 1, 1],
          [0, 1, 0, 0],
          [0, 0, 0, 0]],

         [[0, 1, 1, 0],
          [0, 1, 0, 1],
          [0, 1, 0, 1]]])
    >>> mask.shape
        [2,3,4]
    >>> consecutive_masking(mask, min_event_length=2, axis=1, dtype=int)
    >>> array(
      [[[0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]],

       [[0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 1]]]
       )
    # Workflow inside function
    # First for loop
    #     >> mask_index @ index = 0 :
    #         [[[1, 1, 1, 1],
    #           [0, 1, 0, 0]],
    #          [[0, 1, 1, 0],
    #           [0, 1, 0, 1]]])
    #     >> mask_index @ index = 1 :
    #         [[[0, 1, 0, 0],
    #           [0, 0, 0, 0]],
    #          [[0, 1, 0, 1],
    #           [0, 1, 0, 1]]]
    # >>> temp
    #     [[[0, 1, 0, 0],
    #       [0, 0, 0, 0]],
    #      [[0, 1, 0, 0],
    #       [0, 1, 0, 1]]]
    # >>> z
    #     [[[0, 0, 0, 0]]]
    # Second for loop :
    #     >> temp_append  @ index = 0 :
    #         [[[0, 1, 0, 0],
    #           [0, 0, 0, 0],
    #           [0, 0, 0, 0]],
    #          [[0, 1, 0, 0],
    #           [0, 1, 0, 1],
    #           [0, 0, 0, 0]]])
    #     >> temp_append @ index = 1 :
    #         [[[0, 0, 0, 0],
    #           [0, 1, 0, 0],
    #           [0, 0, 0, 0]],
    #          [[0, 0, 0, 0],
    #           [0, 1, 0, 0],
    #           [0, 1, 0, 1]]]
    # >>> result
    #     [[[0, 1, 0, 0],
    #       [0, 1, 0, 0],
    #       [0, 0, 0, 0]],
    #      [[0, 1, 0, 0],
    #       [0, 1, 0, 1],
    #       [0, 1, 0, 1]]]
    """

    # Change dtype of mask into bool
    mask = mask.astype(bool)

    # ------
    # PROBLEMS WITH TESTING
    # check if input mask is numpy.ndarray:
    # if not isinstance(mask, np.ndarray) :
    #     raise TypeError("Mask is not a numpy.ndarray!")
    # ------

    # ------
    # TODO: need to make sure this is not a mutable return!
    # ------
    # In simple case of min_event_length, mask can be directly returned
    if min_event_length == 1:
        return mask.astype(dtype)

    # Get shape of mask along wanted axis. Used later slicing operations.
    mask_len = mask.shape[axis]
    # min_event_length needs to be legal
    if min_event_length > mask_len:
        raise ValueError(
            f"min_event_length : {min_event_length} exceeds mask length : {mask_len} along axis {axis}!"
        )

    # Create array `temp` with reduced length along axis :
    # mask_len - min_event_length = 1
    for index in range(min_event_length):

        # create indices for take operation with len along axis : mask_len - min_event_length = 1
        indices = np.arange(index, mask_len - min_event_length + index + 1, 1)
        # get mask sub-arrays
        mask_index = np.take(mask, indices=indices, axis=axis)
        # initialize temp in first iteration
        if index == 0:
            temp = mask_index
        # Use `&` operator when combining mask sub-arrays
        temp = temp & mask_index

    # Now `temp` needs to be appended with False values to have same dimension as `mask`.
    # Appending depends on position along axis determined by index in the for loop.

    # Axis for which shape of `mask` and `temp` differs, the difference will be used for as dimension len, if difference equls 0, corresponding shape of `mask` is used.
    # Yields to np.concatenate([temp, z], axis).shape == mask.shape. -> Desired output shape is reached.
    diff = np.array(mask.shape) - np.array(temp.shape)
    diff[diff == 0] = np.array(mask.shape)[diff == 0]
    z = np.zeros(diff)

    # Following suggestion of `Asclepius` from stackoverflow to use numpy full, to make sure esult will be fully False
    # initialize result as all False
    result = np.full(mask.shape, False)
    for index in range(min_event_length):
        # at first index: append z to end of temp
        if index == 0:
            temp_append = np.concatenate((temp, z), axis=axis)
        # at last index: append z to beginning of temp
        elif index == min_event_length - 1:
            temp_append = np.concatenate((z, temp), axis=axis)
        else:
            z_split = np.split(
                z, [index], axis=axis
            )  ## IMPORTANT: Index need to be put as list
            temp_append = np.concatenate((z_split[0], temp, z_split[1]), axis=axis)
        # + operator will be used to identifiy each day ehich is part of the heat wave (see Note)
        result = result + temp_append

    return result.astype(dtype)


# TODO decide for a el nino and la nina definition
def el_nino(d, threshold=1, min_event_length=5, axis=0, dtype=bool):
    """Returns mask of El Nino events with given threshold and minimal event duration for a given input data.

    Parameters
    ----------
    d: numpy.ndarray
        Input data containing ENSO index values.
        Default to 1
    min_event_length: int
        Minimal event length. (Greater equal)
        Default to 5
    axis : int, optional
        Axis along which event needs to exist.
        If `axis` is not specified, it defaults is 0.
    dtype : dtype, optional
        Type of the returned array.
        If `dtype` is not specified, it defaults is `bool` dtype.
    Returns
    -------
    numpy.ndarray
        Array with True for all indices part of an El Nino event defined by the parameters.

    """

    mask = np.greater_equal(d, threshold)
    return consecutive_masking(
        mask=mask,
        min_event_length=min_event_length,
        axis=axis,
        dtype=dtype,
    )


def la_nina(d, threshold=-1, min_event_length=5, axis=0, dtype=bool):
    """Returns mask of La Nina events with given threshold and minimal event duration for a given input data.

    Parameters
    ----------
    d: numpy.ndarray
        Input data containing ENSO index values.
        Default to 1
    min_event_length: int
        Minimal event length. (Greater equal)
        Default to 5
    axis : int, optional
        Axis along which event needs to exist.
        If `axis` is not specified, it defaults is 0.
    dtype : dtype, optional
        Type of the returned array.
        If `dtype` is not specified, it defaults is `bool` dtype.
    Returns
    -------
    numpy.ndarray
        Array with True for all indices part of a La Nina event defined by the parameters.

    """
    mask = np.less_equal(d, threshold)

    return consecutive_masking(
        mask=mask,
        min_event_length=min_event_length,
        axis=axis,
        dtype=dtype,
    )
