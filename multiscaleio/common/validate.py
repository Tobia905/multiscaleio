from __future__ import annotations

import pandas as pd
import numpy as np
import warnings
from typing import Union, Iterable, Optional

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
    array = check_array(array)
    return (
        array[:, None] if len(array.shape) == 1 else array
    )


def check_index(df: pd.DataFrame):
    if df.index.tolist() != np.arange(len(df.index)).tolist():
        warnings.warn(
            "Index is not range(0, len(df)),"
            "this may cause problems when "
            "performing transormations "
            "routines. Better use reset_index."
        )


def check_feature_names(
    array: DataType, feature_names: Optional[Iterable[str or int]] = None
) -> np.array:

    return (
        get_feature_names_in(array) 
        if feature_names is None else np.array(feature_names)
    )


def separete_date_col(
    array: DataType, feature_names: Iterable, date_col: str
) -> tuple[np.ndarray, np.ndarray]:

    array = check_array(array)
    date_idx = check_date_col(feature_names, date_col)
    date = array[:, date_idx]
    X = np.delete(array, date_idx, axis=1)

    return X, date


def check_date_col_in_features(feature_names: Iterable, date_col: str = None):
    if date_col:
        date_idx = np.argwhere(feature_names == date_col)
        feats_to_look = np.delete(feature_names, date_idx)

    else:
        feats_to_look = feature_names

    return feats_to_look