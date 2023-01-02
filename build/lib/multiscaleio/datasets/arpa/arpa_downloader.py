from __future__ import annotations

from typing import Iterable, Union, Callable
from multiscaleio.datasets.arpa.config_arpa import ArpaConfig
from joblib import Parallel, delayed
import re
import pandas as pd
import warnings

CACHE = {}

def __cached(func: Callable[...]) -> Callable[...]:
    """
    Decorator to store downloaded df in cache.

    args:
        func (Callable): function to decorate.

    returns:
        df (Callable): decorated function.
    """
    def wrapper(*args, **kwargs):
        url = kwargs.get("data_url", "")
        if url in CACHE.keys():
            df = CACHE[url]

        else:
            df = func(*args, **kwargs)
            CACHE[url] = df
        return df

    return wrapper


def _uniform_column_names(columns: list[str]) -> list[str]:
    """
    Uniforms column names for both arpa sensors data
    and metadata.

    args:
        columns (list): columns to uniformize.

    returns:
        list: uniformed columns.
    """
    columns = list(
        map(
            lambda col: re.sub(
                "nometiposensore", "tipologia", col, flags=re.IGNORECASE
            ),
            columns
        )
    )
    return [
        re.sub("\sdi", "", col.lower()).replace("à", "a") 
        for col in columns
    ]


def _uniform_dates(dates: Iterable[str]) -> Iterable[str]:
    """
    Uniform dates reported as strings so that they
    can be turned to datetime objects.

    args:
        dates (Iterable): dates to uniformize.

    returns:
        Iterable: processed dates.
    """
    is_obj = True 
    if isinstance(dates, pd.Series):
        is_obj = False if dates.dtype != "object" else True

    elif not(
        all(isinstance(date, "str") for date in filter(dates, None))
    ):
        is_obj = False

    if not is_obj:
        warnings.warn(
            "Dates are not in string format, so they "
            "will not be uniformed. If this will cause "
            "errors or unexpected behaviours, turn them "
            "into strings before running the function again."
        )
        dates = dates
    else:
        dates = [date.split(".")[0] for date in dates]

    return dates


@__cached
def preprocess_arpa(
    meta: pd.DataFrame, 
    data_url: str = "", 
    aggfunc: Union[str, list[str]] = "mean",
    compression: str = "zip",
    **kwargs
) -> pd.DataFrame:
    """
    Retrieves ARPA data from Lombardy region site. Sensors 
    measurements are turned into structured dataframes 
    by using stations metadata to extract the measured quantity, 
    and using these as columns. 
    Finally, the dataframe is resampled by day, aggregated if
    aggfunc is not None and merged with geographical data.

    args:
        meta (pd.DataFrame): stations metadata.
        data_url (str): url of data.
        aggfunc (str): aggregation function.
        compression (str): archive format to be used.
        kwargs: keyword arguments for read_csv from pandas.

    returns:
        final (pd.DataFrame): pivoted dataframe.
    """
    df = pd.read_csv(data_url, compression=compression, **kwargs)
    # column names must be uniformized
    df.columns, meta.columns = (
        _uniform_column_names(df.columns), 
        _uniform_column_names(meta.columns)
    )
    df.data = pd.to_datetime(_uniform_dates(df.data), format=ArpaConfig.DATE_FMT)
    md_columns = [
        "idsensore", "tipologia", "unitamisura", "provincia", "lng", "lat"
    ]
    # get only validated rows
    df = (
        df
        .query("stato=='VA'")
        .merge(meta[md_columns])
        .reset_index(drop=True)
    )
    piv = pd.pivot_table(
        df, index=["data", "provincia"], columns=["tipologia"], values="valore"
    )
    # resampling by day
    final = (
        piv
        .reset_index()
        .groupby("provincia")
        .resample("D", on="data")
    )
    final = final.agg(aggfunc, numeric_only=True) if aggfunc else final
    # getting median longitude and latitude by province
    mean_geo = (
        meta
        .groupby("provincia")
        .agg({"lng": "median", "lat": "median"})
        .reset_index()
    )
    # adding geographical data to final
    final = (
        pd.merge(
            final.reset_index(), 
            mean_geo[['provincia', 'lng', 'lat']], 
            on='provincia'
        )
        .reset_index(drop=True)
    )
    return final

# for meteo data the function breaks the RAM if n_jobs is > 1
def parallel_download(
    meta: pd.DataFrame,
    data_url: Union[str, list[str]] = "",
    pre_dispatch: Union[str, int, float] = "2*n_jobs",
    aggfunc: Union[str, list[str]] = "mean",
    n_jobs: int = 1,
    **kwargs
) -> list[pd.DataFrame]:
    """
    Applies preprocess_arpa in a parallelized setting.

    args:
        meta (pd.DataFrame): stations metadata.
        data_url (str, list): url of data.
        pre_dispatch (str, int, float): args for parallel.
        aggfunc (str, list): aggregation function.
        n_jobs (int): number of cores to use.
        kwargs: generical kwargs for preprocess_data.

    returns:
        out (list): list with one dataframe for each url.
    """
    data_url = (
        [data_url] 
        if not isinstance(data_url, list) 
        else data_url
    )
    if n_jobs > 1 or n_jobs == -1:
        out = Parallel(
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch
        )(
            delayed(preprocess_arpa)(
                meta, data_url=url, aggfunc=aggfunc, **kwargs
            ) 
            for url in data_url
        )
    else: 
        out = [
            preprocess_arpa(
                meta, data_url=url, aggfunc=aggfunc, **kwargs
            )
            for url in data_url
        ]
    return out