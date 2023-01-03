from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, Callable, Optional, Iterable
from multiscaleio.common.validate import (
    check_array, 
    check_array_shape,
    check_feature_names,
    separete_date_col,
    check_date_col_in_features,
    DataType
)
import numpy as np
import pandas as pd


class ReshiftedExpansion(BaseEstimator, TransformerMixin):
    """
    Input data are expanded to include new columns 
    with lagged versions of the original features.
    Use 'window' if you want a range of lags or specify
    a list of lags using 'lags'.

    args:
        window (int): The window that defines the range of lags.
        start (int): Range start.
        step (int): Range step.
        lags (Optional): List of lags. Leave it as None if you
        want to use window.
        keep_t0 (bool): if True, present values are keeped.
        diff (int): order of differentiation of the series.
        date_col (bool, str, int): the name/index of the date 
        column to keep.
        features_names_in_ (Optional, Iterable): features names 
        of input data.
        output_as_df (bool): if True, output of transform is
        returned as a dataframe.

    attributes:
        features_names_out_ (None): features names of output data.
    """
    def __init__(
        self,
        window: int = 1,
        start: int = 0,
        step: int = 1,
        range_step: int = 1,
        lags: Optional[list[int]] = None,
        keep_t0: bool = False,
        diff: int = 0,
        date_col: Union[bool, str, int] = False,
        feature_names_in : Optional[Iterable[str or int]] = None,
        output_as_df: bool = False
    ):
        self.window = window
        self.start = start
        self.step = step
        self.range_step = range_step
        self.lags = lags
        self.keep_t0 = keep_t0
        self.diff = diff
        self.date_col = date_col
        self.feature_names_in_ = feature_names_in
        self.feature_names_out_ = None
        self.output_as_df = output_as_df

    @staticmethod
    def _shift_or_diff(
        array: DataType, 
        order: int=1, 
        op: str = "shift",
        **kwargs
    ) -> np.ndarray:
        """
        Numpy based version of the shift (or diff)
        functions from pandas.

        args:
            array (DataType): input data.
            order (int): order of shifting of diff.
            op (str): the operation to perform.
            kwargs: generical kwargs for np.roll or np.diff.

        returns:
            shifted (np.array): output data.

        raises:
            ValueError: raised if 'op' is not allowed.
        """
        shift_diff = {"shift": np.roll, "diff": np.diff}
        if op not in shift_diff.keys():
            raise ValueError(
                "The selected operation isn't allowed."
               f" Please choose from {shift_diff.keys()}"
            )
        array = check_array(array)
        shifted = shift_diff[op](array, order, axis=0, **kwargs)
        # we need to asses how nans are inserted in data after
        # performing the operation.
        if op == "shift":
            fillna = order
            shifted[:fillna] = np.nan
            
        else:
            fillna = order-1 if order == 1 else order
            shifted = np.insert(shifted, fillna, np.nan, axis=0)
        return shifted

    def fit(self, X: Optional[DataType], y: Optional[DataType] = None):
        return self

    def transform(self, X: DataType, y: Optional[DataType] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        Peforms the shifting expansion: for each feature, a new 
        column representing it's shifted (past or future) version
        is added to data. If 'keep_t0' is True, original features
        are keeped in the output data, while 'date_col' represents
        the column indicating the date.

        args:
            X (DataType): input data.
            y (DataType): unused; added only for coherence
            with sklearn.

        returns:
            shifts (np.ndarray): reshifted data.
        """
        self.feature_names_in_ = check_feature_names(X, self.feature_names_in_)
        if self.window == 0 and len(self.lags) == 0:
            self.feature_names_out_ = self.feature_names_in_
            return X

        else:
            shifts = []
            self.feature_names_out_ = []
            X = check_array(X)
            if self.date_col:
                X, date = separete_date_col(
                    X, self.feature_names_in_, self.date_col
                )

            feats_to_look = check_date_col_in_features(
                self.feature_names_in_, self.date_col
            )
            X = check_array_shape(X)
            for wind in self._shift_range:
                X_shifted = self._shift_or_diff(X, order=wind)
                if wind != 0:
                    # if wind is 0 we don't need to modify the names
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
                # data and feature names need to be concatenated
                # with original ones.
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

        return (
            pd.DataFrame(shifts, columns=self.feature_names_out_)
            if self.output_as_df else shifts
        )

    def get_feature_names_in(self) -> list[Union[int, str]]:
        return self.feature_names_in_.tolist()

    def get_feature_names_out(self) -> list[Union[int, str]]:
        return self.feature_names_out_.tolist()

    @property
    def _shift_range(self) -> Union[range, list[int]]:
        """
        Helper property to set the range of lags.

        returns:
            shift_rng (object, list): The range object or
            periods list.
        """
        if not self.lags:
            shift_rng = (
                range(self.start, self.window, self.step) if self.window > 0 
                else range(self.window, self.start, self.step)
            )
        else:
            shift_rng = self.lags
        return shift_rng


class MultiscaleExpansion(BaseEstimator, TransformerMixin):

    def __init__(
        self, 
        scale: Iterable[int] = [3, 7, 10],
        window_function: Union[str, list[str or Callable], Callable] = "mean",
        date_col: Union[bool, str, int] = False,
        feature_names_in : Optional[Iterable[str or int]] = None,
        output_as_df: bool = False
    ):
        self.scale = scale
        self.window_function = (
            [window_function] 
            if not isinstance(window_function, list)
            else window_function
        )
        self.date_col = date_col
        self.feature_names_in_ = feature_names_in
        self.feature_names_out_ = None
        self.output_as_df = output_as_df

    def fit(self, X: DataType, y: DataType):
        return self

    def transform(self, X: DataType, y: Optional[DataType] = None) -> Union[np.ndarray, pd.DataFrame]:
        return

    def get_feature_names_in(self) -> list[Union[int, str]]:
        return self.feature_names_in_.tolist()

    def get_feature_names_out(self) -> list[Union[int, str]]:
        return self.feature_names_out_.tolist()



