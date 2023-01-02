from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Union, Iterable

DataType = Union[pd.DataFrame, np.ndarray, pd.Series]


def check_array(array: DataType) -> np.ndarray:
    return (
        np.array(array) 
        if not isinstance(array, np.ndarray) 
        else array
    )


def get_feature_names_in(array: DataType) -> np.ndarray:
    return (
        array.columns.values 
        if isinstance(array, pd.DataFrame)
        else np.arange(array.shape[1])
    )


def check_date_col(columns: Iterable, date_col: Union[str, int]) -> int:
    
    columns = list(columns)
    if not isinstance(date_col, str):
        date_col = date_col
    else:
        if columns == np.arange(len(columns)).tolist():
            raise IndexError(
                f"Cannot extract {date_col} from {columns}."
            )
        else:
            date_col = columns.index(date_col)

    return date_col


def check_array_shape(array: DataType) -> np.ndarray:
    array = np.array(array)
    return (
        array[:, None] if len(array.shape) == 1 else array
    )