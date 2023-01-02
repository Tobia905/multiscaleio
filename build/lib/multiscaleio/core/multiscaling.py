from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from tslearn.metrics import dtw
from imblearn.pipeline import Pipeline
from typing import Union, Callable, Optional, Iterable
from multiscaleio.common.validate import (
    check_array, 
    check_date_col, 
    get_feature_names_in,
    check_array_shape,
    DataType
)
import numpy as np
import pandas as pd
import logging
import warnings


class ReshiftedExpansion(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        window: int = 1,
        start: int = 0,
        range_step: int = 1,
        lags: Optional[list[int]] = None,
        keep_t0: bool = False,
        diff: int = 0,
        date_col: Union[bool, str, int] = False
    ):
        self.window = window
        self.start = start
        self.range_step = range_step
        self.lags = lags
        self.keep_t0 = keep_t0
        self.diff = diff
        self.date_col = date_col
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    @staticmethod
    def _shift_or_diff(array: DataType, order: int=1, op: str = "shift", **kwargs) -> np.ndarray:
        shift_diff = {"shift": np.roll, "diff": np.diff}
        array = check_array(array)
        shifted = shift_diff[op](array, order, axis=0, **kwargs)
        if op == "shift":
            fillna = order
            shifted[:fillna] = np.nan
            
        else:
            fillna = order-1 if order == 1 else order
            shifted = np.insert(shifted, fillna, np.nan, axis=0)
        return shifted

    def fit(self, X: Optional[DataType], y: Optional[DataType] = None):
        return self

    def transform(self, X: DataType, y: Optional[DataType] = None) -> np.ndarray:
        self.feature_names_in_ = get_feature_names_in(X)
        if self.window == 0 and len(self.lags) == 0:
            self.feature_names_out_ = self.feature_names_in_
            return X

        else:
            shifts = []
            self.feature_names_out_ = []
            X = check_array(X)
            if self.date_col:
                date_idx = check_date_col(
                    self.feature_names_in_, self.date_col
                )
                date = X[:, date_idx]
                X = np.delete(X, date_idx, axis=1)
                date_idx = np.argwhere(self.feature_names_in_ == self.date_col)
                feats_to_look = np.delete(self.feature_names_in_, date_idx)

            else:
                feats_to_look = self.feature_names_in_

            X = check_array_shape(X)
            for wind in self._shift_range:
                X_shifted = self._shift_or_diff(X, order=wind)
                if wind != 0:
                    self.feature_names_out_.append(
                        [col + f'_shift_{wind}' for col in feats_to_look]
                    )
                shifts.append(X_shifted)
            if self.date_col:
                self.feature_names_out_ = [[self.date_col]] + self.feature_names_out_

            self.feature_names_out_ = np.concatenate(self.feature_names_out_)
            shifts = (
                shifts[1:] 
                if not isinstance(self._shift_range, list) and self.window > 1
                else shifts 
            )
            shifts = np.concatenate(shifts, axis = 1)
            if self.keep_t0:
                to_concat = (
                    (X, shifts) 
                    if self.window > 0 or isinstance(self._shift_range, list) 
                    else (shifts, X)
                )
                shifts = np.concatenate(to_concat, axis = 1)
                concat_names = (
                    (feats_to_look, self.feature_names_out_)
                    if self.window > 0 or isinstance(self._shift_range, list)
                    else (self.feature_names_out_, feats_to_look)
                )
                self.feature_names_out_ = np.append(*concat_names)

            shifts = (
                self._shift_or_diff(shifts, order=self.diff, op="diff") 
                if self.diff > 0 
                else shifts
            )
            if self.date_col:
                shifts = np.insert(shifts, 0, date, axis=1)

        return shifts

    def get_feature_names_in(self) -> list[Union[int, str]]:
        return self.feature_names_in_.tolist()

    def get_feature_names_out(self) -> list[Union[int, str]]:
        return self.feature_names_out_.tolist()

    @property
    def _shift_range(self) -> Union[range, list[int]]:
        if not self.lags:
            shift_rng = (
                range(self.window) if self.window > 0 else 
                range(self.window, self.start, self.stop)
            )
        else:
            shift_rng = self.lags
        return shift_rng
