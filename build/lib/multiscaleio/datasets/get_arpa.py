from __future__ import annotations

from typing import Union
from multiscaleio.datasets.arpa.config_arpa import ArpaConfig
from multiscaleio.datasets.arpa.arpa_downloader import parallel_download
import pandas as pd
import logging

DICT_CONF = ArpaConfig.config_to_dict()

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def get_arpa(
    years: Union[str, list[str]] = list(DICT_CONF["air"].keys())[-1], 
    measure: str = "air",
    pre_dispatch: Union[str, int, float] = "2*n_jobs", 
    aggfunc: str = "mean",
    n_jobs: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Downloads, parses and concats all ARPA data for the
    selected years.

    args:
        years (str, list): years to download.
        measure (str): "air" or "meteo" - kind of ARPA data to obtain.
        pre_dispatch (str, int, float): dispatching strategy for Parallel.
        aggfunc (str): aggregation function for download_and_parse.
        n_jobs (int): cores to use for Parallel.

    returns:
        arpa (pd.DataFrame): final dataframe.
    """
    # an error is raised if measure is not valid
    if measure not in ["air", "meteo"]:
        raise ValueError(
            f"{measure} measure is invalid. Please select "
             "from ('air', 'meteo')."
        )

    # get years from confing
    years = (
        list(DICT_CONF[measure].keys()) if years == "all" else years
    )

    years = [years] if not isinstance(years, list) else years
    period = (
        f"from {sorted(years)[0]} to {sorted(years)[-1]}"
        if len(years) > 1
        else f"for {years[0]}"
    )
    logger.info(f"Retrieving arpa {measure} data {period}")

    links = [DICT_CONF[measure][year] for year in years]
    meta = pd.read_csv(DICT_CONF["metadata"][measure])
    arpa = pd.concat(
        parallel_download(
            meta, data_url = links, pre_dispatch=pre_dispatch, 
            n_jobs=n_jobs, aggfunc=aggfunc, **kwargs
        )
    )
    arpa = (
        arpa.drop(['lng', 'lat'], axis=1) if measure == "meteo" else arpa
    )

    return arpa.reset_index(drop=True)