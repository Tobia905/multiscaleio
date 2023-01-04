from __future__ import annotations

from prophet import Prophet
from multiscaleio.common.prophet_log_suppressor import FitlogSuppressor
from typing import Union
import numpy as np
import pandas as pd
import logging


class ProphetInterpolator:

    def __init__(
        self,
        date_index: Union[str, int] = 0,
        add_sampled_uncertainty: bool = True,
        uncertainty_fit_logs: bool = False,
        **proph_args
    ):

        self.date_index = date_index
        self.add_sampled_uncertainty = add_sampled_uncertainty
        self.uncertainty_fit_logs = uncertainty_fit_logs
        # we need to take track of calls to transform method
        self.transformations = 0
        # disable the prophet initialization infos
        logging.getLogger('prophet').setLevel(logging.WARNING) 
        self.prophet = Prophet(**proph_args)

        