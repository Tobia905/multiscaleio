from __future__ import annotations

from multiscaleio.common.validate import (
    DataType, 
    check_index, 
    check_pandas_nan, 
    check_array,
    display_nan_warning
)
from typing import Union, Optional, Callable
from math import floor, modf
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_breuschpagan
from functools import partial
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def insert_nan_into_window(array: DataType, window_size: int) -> np.ndarray:
    return np.insert(
        array, 0, np.repeat(np.nan, window_size-1)
    )


def get_statistics(stat: str):
    np_stat_funcs = {
        "mean"  : np.nanmean,
        "median": np.nanmedian,
        "std"   : np.nanstd,
        "min"   : np.nanmin,
        "max"   : np.nanmax
    }
    return (
        np_stat_funcs[stat] if stat != "all" else np_stat_funcs
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


def moving_average(array: DataType, window: np.ndarray, **kwargs) -> np.ndarray:
    """
    Full numpy version of moving average. The function
    is thought to be used inside the 'rolling' function.

    args:
        array (DataType): input data.
        window (np.ndarray): window array.

    returns:
        np.ndarray: output data.
    """
    array = check_pandas_nan(array, **kwargs)
    ma = np.divide(
        np.convolve(array, window, "valid"), len(window)
    )
    ma = insert_nan_into_window(ma, len(window))
    return np.array(ma, dtype=float)


def base_moving_function(
    array: DataType, 
    window: int, 
    func: str = "median",
    **kwargs
) -> np.ndarray:
    """
    Full numpy version of moving functions. Thought to be 
    used inside the 'rolling' function.

    args:
        array (DataType): input data.
        window (int): window size.
        kwargs: kwargs for check_pandas_nan.

    returns:
        np.ndarray: output data.
    """
    # this is needed since numpy doesn't recognise some
    # nans in pandas Serieses.
    assert func != "all"
    array = check_pandas_nan(array, **kwargs)
    ms = get_statistics(func)(
        sliding_window_view(array, window), axis=-1
    )
    return insert_nan_into_window(ms, window)


def rolling(
    array: DataType, 
    *win_args, 
    window: int = 7, 
    func: Union[str, Callable] = "mean", 
    win_type: str = "ones",
    **kwargs
) -> np.ndarray:
    """
    Handler of pre-defined (or custom-made) window
    functions.

    args:
        win_args: arguments for scipy.signal.get_window
        window (int): size of the window.
        func (str, Callable): the function to apply.
        win_type (str): the window type for moving_average. 
        kwargs: kwargs for check_pandas_nan.

    returns:
        (np.ndarray): output data.
    """
    allfunctions = get_statistics("all").keys()
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
        moving = moving_average

    elif func != "mean" and func in allfunctions.keys():
        window = window
        moving = partial(base_moving_function, func=func)

    return (
        moving(array, window, **kwargs) 
        if func in allfunctions else func(array, window)
    )


def ts_hsched_test(array: DataType, **kwargs):
    """
    Performs the Breusch-Pagan homoschedasticity
    test in a time series fashion.

    args:
        array (DataType): input data.
        **kwargs: keyword arguments for het_breuschpagan.

    returns:
        tuple: BP test results.
    """
    array = check_array(array)
    if np.nansum(array) / len(array) > .35:
        display_nan_warning(array)

    array = np.nan_to_num(array, np.nanmedian(array))
    array = array[:, None] if len(array.shape) == 1 else array
    # concatenate output data with an array
    # representing timestamps
    array = np.concatenate(
        (np.arange(len(array)).reshape(-1, 1), array), axis=1
    )
    # obtaining OLS residuals
    lr = LinearRegression()
    X, y = array[:, 0].reshape(-1, 1), array[:, 1]
    _ = lr.fit(X, y)
    fitted = lr.predict(X)
    residuals = array[:, 1] - fitted
    # bp test requires a 2-dim array for input data
    bp_data = np.concatenate(
        (
            (np.repeat(1, len(array))[:, None]), 
            array[:, 1][:, None]
        ),
        axis=1
    )
    return het_breuschpagan(residuals, bp_data, **kwargs)


def auto_seasonality(array: DataType, **kwargs):
    """
    Extract the length of the seasonality period
    using power spectral density, computed using 
    the Welch's method. When applied in the Prophet
    interpolator, the seasonality componenta
    is automatically extracted from the trainin set 
    (not nans). This means that, for long periods of 
    seasonality (e.i. a year), the performances of 
    the interpolator heavily rely on the % of not 
    nans in the series.

    args:
        array: input data.
        kwargs: keyword arguments for scipy welch function.

    returns:
        float: seasonality period.
    """
    array = check_array(array)
    array = array[~np.isnan(array)]
    arr_spectrum = signal.welch(array, **kwargs)
    return 1 / arr_spectrum[0][np.argmax(arr_spectrum[1])]