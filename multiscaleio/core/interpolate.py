from __future__ import annotations

from prophet import Prophet
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from multiscaleio.common.validate import DataType, check_index
from multiscaleio.core.sample_interval import UncertaintySampler
from multiscaleio.core.interpolate import ts_hsched_test
from typing import Union, Optional
import numpy as np
import pandas as pd
import logging
import warnings


class ProphetInterpolator(BaseEstimator, TransformerMixin):
    """
    Interpolates both time related missing values, coming from 
    shifting process and time series operations such as 
    convolutions (eg. moving averages), and random ones. 
    Interpolations are performed using Facebook's Prophet model,
    a method belonging to the class of Generalized Additive Models.
    Informations about Prophet can be found at the following url: 
    https://github.com/facebook/prophet.
    
    args:
        date_index (str): Date column to use for indexing 'ds'.
        and 'y' (see Prophet).
        add_sampled_uncertainty (bool): if true, error distribution is 
        fitted using training data and added to predictions.
        uncertainty_fit_logs (bool): if True, logs from Fitter are
        displayied.
        output_as_series (bool): if True, output is returned as a
        pd.Series.
        disable_prophet_logs (bool): if True, fit and init logs from
        Prophet are disabled.
        uncertainty_fitter_timeout (int): timeout - for Fitter object
        - after which the fitting of a distribution is skipped.
        proph_args: Prophet keyword arguments.
        
    attributes:
        original_feature_name (str): Original name of the Series.
        prophet (Prophet): initialized Prophet model.
        transformations (int): Indicates the calls to transform
        method.
    """
    def __init__(
        self,
        date_index: Union[str, int] = 0,
        add_sampled_uncertainty: bool = False,
        uncertainty_fit_logs: bool = False,
        output_as_series: bool = False,
        disable_prophet_logs: bool = True,
        uncertainty_fitter_timeout: int = 30,
        **proph_args
    ):
        self.date_index = date_index
        self.add_sampled_uncertainty = add_sampled_uncertainty
        self.uncertainty_fit_logs = uncertainty_fit_logs
        # we need to take track of calls to transform method
        self.transformations = 0
        self.output_as_series = output_as_series
        # disable the prophet initialization infos
        logging.getLogger('prophet').setLevel(logging.WARNING) 
        self.prophet = Prophet(**proph_args)
        self.original_feature_name = None
        self.disable_prophet_logs = disable_prophet_logs
        self.uncertainty_fitter_timeout = uncertainty_fitter_timeout

    def fit(self, X: DataType, y: Optional[DataType] = None):
        """
        Creates training set using non missing values from
        X and fits the Prophet model on it.

        args:
            X (DataType): input data.
            y (DataType): unused; added only for coherence
            with sklearn.

        returns:
            self (ProphetInterpolator): fitted instance of 
            ProphetInterpolator. 

        raises:
            ValueError: length of the shape of X is equal
            to one.
            ValueError: shape of X is greater than 2.
        """
        X = X.copy()
        if len(X.shape) == 1:
            raise ValueError(
                "Data should be two dimensional, with "
                "one column indicating the date and one "
                "indicating the feature to interpolate."
            )
        else:
            if X.shape[1] != 2:
                raise ValueError(
                    "The interpolator is tougth to be "
                    "monovariate. Make sure that X includes "
                    "a date column and a feature one."
                )
        if isinstance(X, pd.DataFrame):
            check_index(X)

        else:
            # input data should be a dataframe to be 
            # used in the Prophet model
            X = pd.DataFrame(X)
            if isinstance(self.date_index, str):
                raise TypeError(
                    f"{type(self.date_index)} date index type is not "
                     "allowed for {X.columns}"
                )
        self.original_feature_name = [
            X[col].name for col in X.columns if col != self.date_index
        ]
        # Prophet requires that feature names are "ds" and "y"
        # and that "ds" is a DatetimeIndex.
        X.columns = ["ds", "y"]
        X["ds"] = pd.DatetimeIndex(X["ds"])
        self.train = X[X.y.notna()]

        # disable the prophet fit infos
        if self.disable_prophet_logs:
            logger = logging.getLogger('cmdstanpy')
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
            logger.setLevel(logging.CRITICAL)

        _ = self.prophet.fit(self.train)
            
        self.is_fitted_ = True
        return self

    def sample_uncertainty(
        self, 
        distributions: str = "real_line_support", 
        size: int = 100, 
        **kwargs
    ) -> np.ndarray:
        """
        Fits a list of distributions on deviations from yhat and 
        training data, automatically chooses the best fitting one
        and samples from that one using the fitted parameters.

        args:
            distributions (str, list): a list of distributions or a string 
            indicating some subsample of distributions.
            size (int): the number of samples to draw.
            **kwargs: generical kwargs from the FitterSampler object.

        returns:
            np.ndarray: samples extracted from the best fitting distribution.
        """
        check_is_fitted(self)
        # adding random oscillations to interpolated values 
        # can lead to unwanted results if variance is
        # heteroschedastic.
        if ts_hsched_test(self.train)[1] < 0.05:
            warnings.warn(
                f"{self.original_feature_name} has "
                 "heteroschedastic variance, so adding "
                 "random uncertainty based on residual distribution "
                 "could lead to unexpected results. Better set "
                 "add_sample_uncertainty to False."
            )
        fitted = self.prophet.predict(self.train)
        err = self.train["y"].reset_index(drop=True) - fitted["yhat"]
        # the best fitting real line supported distribution
        # is fitted on the errors between Prophet predictions
        # and real values.
        fs = UncertaintySampler(
            err.values, 
            distributions=distributions, 
            fit_logs=self.uncertainty_fit_logs, 
            **kwargs
        )
        return fs.sample(not_fitted_action="fit", size=size)

    def transform(
        self, 
        X: DataType, 
        y: Optional[DataType] = None, 
        **kwargs
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Performs the interpolation. Missing Values are predicted
        according to Prophet's fitted parameters. If add uncertainty
        is True in the init, random oscillations will be added to
        the predictions, according to the best fitting distribution.
        
        args:
            X (DataType): Series with values to interpolate and date.
            y (DataType): unused; added only for coherence
            with sklearn.
            
        returns:
            pd.Series: Series with interpolated missing values.
        """
        check_is_fitted(self)
        X = X.copy()
        X.columns = ["ds", "y"]
        X["ds"] = pd.DatetimeIndex(X["ds"])

        # hold out set are nans
        test = X[X["y"].isna()][["ds"]]
        # we also need to store not nans if the transform
        # method is performed on the test set
        test_notna = X[X["y"].notna()]
        interp = self.prophet.predict(test)[["ds", "yhat"]]
        self.transformations += 1

        # random oscillations are added to the predictions
        # if add_sampled_uncertainty is True
        if self.add_sampled_uncertainty:
            samples = self.sample_uncertainty(size=len(interp), **kwargs)
            interp["yhat"] = interp["yhat"] + samples 

        interp = interp[["ds", "yhat"]]
        to_conc = (
            self.train 
            if self.transformations == 1 else test_notna
        )
        modeled = pd.concat((to_conc, interp[["ds", "yhat"]]), axis = 0).fillna(0)
        modeled[self.get_original_feature_name()] = modeled["y"] + modeled["yhat"]
        modeled = (
            modeled
            .sort_values("ds")
            .reset_index(drop = True)
            [self.get_original_feature_name()]
        )
        modeled.index = (
            X.index if self.transformations > 1 else modeled.index
        )
        return modeled if self.output_as_series else np.array(modeled)

    def get_original_feature_name(self):
        return self.original_feature_name[0]