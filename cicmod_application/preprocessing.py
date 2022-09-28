import numpy as np

def split_sequence(sequence, n_steps):
    """Split time series into samples of specified length.
    
    Parameters
    ----------
    sequence: numpy.ndarray
        Sequence or time series to be split with dimensions (time steps, features).
    n_steps: int
        Length of samples after splitting.

    Returns
    -------
    numpy.ndarray
        Split sequence or time series with dimensions (samples, n_steps, features).
        
    """
    X = list()
    for i in range(len(sequence)):
        
        # Find the end of this pattern
        end_ix = i + n_steps
        
        # Check if we are beyond the sequence
        if end_ix > len(sequence):
            break
        
        # Gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        X.append(seq_x)
    
    return np.array(X)


