import numpy as np
import numbers


def check_random_state(seed):
    """
    Create RandomState for the seed.
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If None, return a new RandomState singleton
        If int, return RandomState using seed
        if RandomState, return seed

    Returns
    -------
    np.random.RandomState
    """
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    else:
        ValueError(f'can not be used as random state instance {seed} ')


def norm(X, ord_=2, squared=True):
    """

    Parameters
    ----------
    X : array, shape (i, j)
    ord_ : int, {1,2}
        When 1, Manhattan distance metric is applied; return Summation abs
        When 2, Euclidean distance metric is applied; return Summation squared
    squared : bool, default True
        If ord_=2
            If True, return squared
            If False, return not squared

    Returns
    -------

    """
    if ord_ == 1:
        return np.einsum('ij->i', np.abs(X))
    elif ord_ == 2:
        _norm = np.einsum('ij,ij->i', X, X)
        return _norm if squared else np.sqrt(_norm)
    else:
        raise ValueError(f'Invalid norm <ord_> value, supported [1,2], provided: {ord_}')
