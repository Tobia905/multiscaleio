from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

@dataclass
class ArpaConfig:

    AIR = {
        "2015": "https://www.dati.lombardia.it/download/bpin-c7k8/application%2Fzip",
        "2016": "https://www.dati.lombardia.it/download/7v3n-37f3/application%2Fzip",
        "2017": "https://www.dati.lombardia.it/download/fdv6-2rbs/application%2Fzip",
        "2018": "https://www.dati.lombardia.it/download/4t9j-fd8z/application%2Fzip",
        "2019": "https://www.dati.lombardia.it/download/j2mz-aium/application%2Fzip",
        "2020": "https://www.dati.lombardia.it/download/88sp-5tmj/application%2Fzip",
        "2021": "https://www.dati.lombardia.it/download/wzmx-9k7n/application%2Fzip"
    }

    METEO = {
        "2015": "https://www.dati.lombardia.it/download/knr4-9ujq/application%2Fzip",
        "2016": "https://www.dati.lombardia.it/download/kgxu-frcw/application%2Fzip",
        "2017": "https://www.dati.lombardia.it/download/vx6g-atiu/application%2Fzip",
        "2018": "https://www.dati.lombardia.it/download/sfbe-yqe8/application%2Fzip",
        "2019": "https://www.dati.lombardia.it/download/wrhf-6ztd/application%2Fzip",
        "2020": "https://www.dati.lombardia.it/download/erjn-istm/application%2Fzip",
        "2021": "https://www.dati.lombardia.it/download/49n9-866s/application%2Fzip"
    }

    META_AIR = "https://www.dati.lombardia.it/api/views/ib47-atvt/rows.csv?accessType=DOWNLOAD"
    META_MET = "https://www.dati.lombardia.it/api/views/nf78-nj6b/rows.csv?accessType=DOWNLOAD"
    DATE_FMT = "%d/%m/%Y %H:%M:%S"

    @classmethod
    def config_to_dict(cls) -> Dict[str, Union[Dict[str, str], str]]:
        return {
            "air"  : cls.AIR,
            "meteo": cls.METEO,
            "metadata" : {
                "air"  : cls.META_AIR,
                "meteo": cls.META_MET
            },
            "time_format": cls.DATE_FMT
        }