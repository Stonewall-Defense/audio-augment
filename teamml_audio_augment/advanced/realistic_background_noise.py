###############################################################################
# Global Imports
###############################################################################
from enum import Enum
import os
from pathlib import Path
from typing import Literal, Optional

###############################################################################
# 3PP Imports
###############################################################################
import pandas as pd

###############################################################################
# Local Imports
###############################################################################
from teamml_audio_augment.augmentations.add_background_noise import AddBackgroundNoise


###############################################################################
# Enums
###############################################################################
class BackgroundNoise(Enum):
    AMBIENT = "ambient"
    ANIMAL = "animal"
    HUMAN = "human"
    VEHICLE = "vehicle"
    WEATHER = "weather"


BG_NOISE_TYPE = Literal[
    "ambient",
    "animal",
    "human",
    "vehicle",
    "weather",
]


###############################################################################
# ! EXPORTS
###############################################################################
def make_background_noise_source(path_to_dataset: str | Path,
                                 sound_class: BackgroundNoise | BG_NOISE_TYPE,
                                 *,
                                 sample_rate: Optional[int] = None,
                                 min_snr_db: float = 3.0,
                                 max_snr_db: float = 20.0,
                                 p: float = 0.5,
                                 ):
    sc = sound_class.value if isinstance(sound_class, BackgroundNoise) else sound_class
    fq_metadata_file = os.path.join(path_to_dataset, "metadata.csv")
    metadata = pd.read_csv(fq_metadata_file)
    filtered = [os.path.join(path_to_dataset, "data", f) for f in list(metadata.loc[metadata["class_name"] == sc]["filename"])]
    return AddBackgroundNoise(filtered, sample_rate=sample_rate, min_snr_db=min_snr_db, max_snr_db=max_snr_db, p=p)
