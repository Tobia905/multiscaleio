import scipy
import numpy as np
import logging
from fitter import Fitter
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


def real_line_supported() -> list[str]:
    return [
        "alpha",
        "norm",
        "cauchy",
        "laplace",
        "t",
        "logistic",
        "norminvgauss",
        "gumbel_l",
        "gumbel_r",
        "hypsecant"
    ]
    

class UncertaintySampler(Fitter):
    """
    An extention of the Fitter object from fitter package:
    https://fitter.readthedocs.io/en/latest/. The class inherits
    from fitter and adds a sampler method, which draws samples 
    from the best fitting distribution.

    args:
        args: generic args from Fitter class.
        kwargs: generic kwargs from Fitter class.

    attributes:
        is_fitted (bool): False if the object has to be fitted,
        True otherwise.
        distributions (list): list of selected distributions. All
        scipy's distributions will be used if is set to None; the
        most common will be used if is set to 'common' and the ones 
        supported on the whole real line will be used  if is set to 
        'real_line_support'.
    """
    def __init__(self, *args, fit_logs: bool = False, **kwargs):

        super(UncertaintySampler, self).__init__(*args, **kwargs)
        self.distributions = kwargs.get("distributions", None)
        self.fit_logs = fit_logs

        if self.distributions == "real_line_support":
            self.distributions = real_line_supported()

    def fit(self):
        """
        Same method from original Fitter class, extended with
        the attribute 'is_fitted_', which is set to True if 
        the class is fitted.
        
        returns:
            fitted instance of the classe.
        """
        if not self.fit_logs:
            logger = logging.getLogger()
            logger.disabled = True

        _ = super().fit()
        self.is_fitted_ = True
        return self

    def sample(self, not_fitted_action: str = "raise", size: int = 100) -> np.ndarray:
        """
        Main extention to the Fitter object. Draws samples 
        from the best fitting distributions parameters
        fitted through log-likelihood maximization.

        args:
            not_fitted_action (str): the action to undertake
            if the class has not been fitted before sampling.
            default: 'raise'.

        returns:
            np.ndarray: array with drawn samples.
        
        raises:
            NotFittedError if the class if not fitted and 
            not_fitted_action is set to 'raise'.
        """
        try:
            check_is_fitted(self)

        except NotFittedError as error:
            if not_fitted_action == "raise":
                raise error

            else:
                _ = self.fit()

        best_dist = list(self.get_best().keys())[0]
        params = self.fitted_param[best_dist]
        return eval("scipy.stats."+best_dist+".rvs")(*params, size=size)