from __future__ import annotations

from typing import Union, Optional
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
    """Downloads, parses and concats all ARPA data for the
       selected years.

       args:
           years (str, list): years to download.
           kind (str): "air" or "meteo" - kind of ARPA data to obtain.
           pre_dispatch (str, int, float): dispatching strategy for Parallel.
           aggfunc (str): aggregation function for download_and_parse.
           subset (str): must be a boolean query for df.query.
           n_jobs (int): cores to use for Parallel.

       returns:
           pd.DataFrame: final dataframe.
    """
    if measure not in ["air", "meteo"]:
        raise ValueError(
            f"{measure} measure is invalid. Please select "
             "from ('air', 'meteo')."
        )

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

    # links = [AIR_MET[measure]["data"][year] for year in years]
    # meta = pd.read_csv(AIR_MET[measure]["meta"])
    # air = parallel_download(
    #     links, meta, pre_dispatch=pre_dispatch, 
    #     n_jobs=n_jobs, aggfunc=aggfunc, **kwargs
    # )
    # total = pd.concat(air).reset_index(drop=True)

    # if measure == "meteo":
    #     total = total.drop(['lng', 'lat'], axis=1)

    return #total