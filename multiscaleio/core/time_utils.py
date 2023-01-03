from __future__ import annotations

from multiscaleio.common.validate import DataType, check_index
from typing import Union, Optional
from math import floor, modf
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
import pandas as pd
import numpy as np


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
    ma = np.divide(
        np.convolve(array, window, "valid"), len(window)
    )
    ma = insert_nan_into_window(ma, len(window))
    return np.array(ma, dtype=float)


def moving_median(array: DataType, window: int) -> np.ndarray:
    ms = np.median(
        sliding_window_view(array, window), axis=-1
    )
    return insert_nan_into_window(ms, window)


def rolling(
    array: DataType, 
    *win_args, 
    window: int = 7, 
    func: str = "mean", 
    win_type: str = "ones"
) -> np.ndarray:

    rf = {
        "mean": moving_average, "median": moving_median
    }
    if func == "mean":
        window = (
            np.ones(window) 
            if win_type == "ones" 
            else signal.get_window((win_type, *win_args), window)
        )
    else:
        window = window

    return rf[func](array, window)