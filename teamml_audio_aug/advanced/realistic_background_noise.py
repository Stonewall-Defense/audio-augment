###############################################################################
# Global Imports
###############################################################################
from enum import Enum
import os

###############################################################################
# 3PP Imports
###############################################################################
from audiomentations import AddBackgroundNoise
import pandas as pd


###############################################################################
# Enums
###############################################################################
class BackgroundNoise(Enum):
    AMBIENT = "ambient"
    ANIMAL = "animal"
    HUMAN = "human"
    VEHICLE = "vehicle"
    WEATHER = "weather"


###############################################################################
# ! EXPORTS
###############################################################################
def make_background_noise_source(path_to_dataset: str,
                                 sound_class: BackgroundNoise,
                                 min_snr_db: float = 3.0,
                                 max_snr_db: float = 20.0,
                                 p: float = 0.5,
                                 ):
    fq_metadata_file = os.path.join(path_to_dataset, "metadata.csv")
    metadata = pd.read_csv(fq_metadata_file)
    filtered = [os.path.join(path_to_dataset, "data", f) for f in list(metadata.loc[metadata["class_name"] == sound_class.value]["filename"])]
    return AddBackgroundNoise(filtered, min_snr_db=min_snr_db, max_snr_db=max_snr_db, p=p)
