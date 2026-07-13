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
from teamml_audio_augment.augmentations.apply_impulse_response import ApplyImpulseResponse


###############################################################################
# Enums
###############################################################################
class RirLocation(Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


RIR_LOC = Literal[
    "indoor",
    "outdoor",
]


###############################################################################
# Helpers
###############################################################################
def _filter(metadata: pd.DataFrame, location: RirLocation | RIR_LOC):
    loc = location.value if isinstance(location, RirLocation) else location
    return [f for f in list(metadata.loc[metadata["class_name"] == loc]["filename"])]


###############################################################################
# ! EXPORTS
###############################################################################
def make_location_rir(path_to_dataset: str | Path,
                      rir_location: RirLocation | RIR_LOC,
                      *,
                      sample_rate: Optional[int] = None,
                      p: float = 0.5,
                      cahce_size: int = 128,
                      ):
    fq_metadata_file = os.path.join(path_to_dataset, "metadata.csv")
    metadata = pd.read_csv(fq_metadata_file)
    filtered = [os.path.join(path_to_dataset, "data", f) for f in _filter(metadata, rir_location)]
    return ApplyImpulseResponse(filtered, sample_rate=sample_rate, p=p, lru_cache_size=cahce_size)
