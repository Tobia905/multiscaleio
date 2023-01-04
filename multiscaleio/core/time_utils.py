from __future__ import annotations

from multiscaleio.common.validate import DataType, check_index, check_pandas_nan
from typing import Union, Optional, Callable
from math import floor, modf
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def insert_nan_into_window(array: DataType, window_size: int) -> np.ndarray:
    return np.insert(
        array, 0, np.repeat(np.nan, window_size-1)
    )


def ts_train_test_split(
    X: DataType, 
    y: Optional[Union[DataType, list]] = None, 
    test_size: float = .20
) -> DataType:
    """
    Train-test split for time series data: splitting
    is sequential and data are not shuffled.

    args:
        X (DataType): input data to be splitted.
        y (DataType, list): output data to be splitted.
        test_size (float): size of the split.

    returns:
        tuple: splitted data.
    """
    type_checks = (pd.DataFrame, pd.Series)
    # check if indexes are in the right orders.
    try:
        for data in [X, y]:
            if isinstance(data, type_checks):
                check_index(data)

    except AttributeError:
        if isinstance(X, type_checks):
            check_index(X)
        
    idx = len(X) * test_size
    # idx cannot be a float
    idx = floor(idx) if modf(idx)[0] < 0.5 else round(idx)
    if y is not None:
        return (
            X[:-int(idx)], X[-int(idx):],
            y[:-int(idx)], y[-int(idx):]
        )
    else:
        return X[:-int(idx)], X[-int(idx):]


def bootstrapped_interval(
    residuals: Union[DataType, list], 
    pred: Union[DataType, list], 
    alpha: float = 0.05, 
    samples: int = 100, 
    to_df: bool = False
) -> Union[pd.DataFrame, tuple]:
    """
    Bootstrapped prediction interval for a generical 
    model's prediction.

    args:
        residuals (DataType, list): predictions residuals.
        pred (DataType, list): actual predictions.
        alpha (float): confidence level.
        samples (int): number of samples to draw.
        to_df (bool): if True, results are turned into a dataframe.

    returns:
        (tuple, pd.DataFrame): upper and lower bounds of CI.
    """
    bootstrap = np.asarray(
        [
            np.random.choice(residuals, size=residuals.shape) 
            for _ in range(samples)
        ]
    )
    q_bootstrap = np.quantile(bootstrap, q=[alpha/2, 1-alpha/2], axis=0)

    pred = np.array(pred)
    y_lower = pred + np.mean(q_bootstrap[0])
    y_upper = pred + np.mean(q_bootstrap[1])
    return (
        (y_lower, y_upper) 
        if not to_df 
        else pd.DataFrame(
            np.array([y_lower, y_upper]).T, columns = ['low', 'up']
        )
    )


def moving_average(array: DataType, window: np.ndarray) -> np.ndarray:
    """
    Full numpy version of moving average. The function
    is thought to be used inside the 'rolling' function.

    args:
        array (DataType): input data.
        window (np.ndarray): window array.

    returns:
        np.ndarray: output data.
    """
    ma = np.divide(
        np.convolve(array, window, "valid"), len(window)
    )
    ma = insert_nan_into_window(ma, len(window))
    return np.array(ma, dtype=float)


def moving_median(array: DataType, window: int) -> np.ndarray:
    """
    Full numpy version of moving median. The function
    is thought to be used inside the 'rolling' function.

    args:
        array (DataType): input data.
        window (int): window size.

    returns:
        np.ndarray: output data.
    """
    ms = np.median(
        sliding_window_view(array, window), axis=-1
    )
    return insert_nan_into_window(ms, window)


def moving_std(array, window: int) -> np.ndarray:
    """
    Full numpy version of moving standard deviation. 
    The function is thought to be used inside the 
    'rolling' function.

    args:
        array (DataType): input data.
        window (int): window size.

    returns:
        np.ndarray: output data.
    """
    # this is needed since numpy doesn't recognise some
    # nans in pandas Serieses.
    array = check_pandas_nan(array)
    ms = np.nanstd(
        sliding_window_view(array, window), axis=-1
    )
    return insert_nan_into_window(ms, window)


def get_window_functions(func: str = "mean"):
    """
    Helper function to get pre-defined window
    functions.

    args:
        func (str): key of the function.

    returns:
        (Callable, dict): selected function or all 
        ones if func is 'all'.
    """
    rollfuncs = {
        "mean"  : moving_average, 
        "median": moving_median,
        "std"   : moving_std
    }
    return (
        rollfuncs[func] if func != "all" else rollfuncs
    )


def rolling(
    array: DataType, 
    *win_args, 
    window: int = 7, 
    func: Union[str, Callable] = "mean", 
    win_type: str = "ones"
) -> np.ndarray:
    """
    Handler of pre-defined (or custom-made) window
    functions.

    args:
        win_args: arguments for scipy.signal.get_window
        window (int): size of the window.
        func (str, Callable): the function to apply.
        win_type (str): the window type for moving_average. 

    returns:
        (np.ndarray): output data.
    """
    allfunctions = get_window_functions("all").keys()
    if func not in allfunctions:
        logger.info(
            f"{func} is not a valid pre-defined rolling "
            "function. If this is intended, just ignore this "
            f"message. Otherwise, select from {allfunctions}"
        )
    if func == "mean":
        window = (
            np.ones(window) 
            if win_type == "ones" 
            else signal.get_window((win_type, *win_args), window)
        )
    else:
        window = window

    return (
        get_window_functions(func)(array, window) 
        if func in allfunctions else func(array, window)
    )