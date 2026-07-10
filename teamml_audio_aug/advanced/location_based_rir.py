###############################################################################
# Global Imports
###############################################################################
from enum import Enum
import os

###############################################################################
# 3PP Imports
###############################################################################
from audiomentations import ApplyImpulseResponse
import pandas as pd


###############################################################################
# Enums
###############################################################################
class RirLocation(Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


###############################################################################
# Helpers
###############################################################################
def _filter(metadata: pd.DataFrame, location: RirLocation):
    if location == RirLocation.INDOOR:
        return [f for f in list(metadata.loc[metadata["class_name"] != RirLocation.OUTDOOR.value]["filename"])]
    else:
        return [f for f in list(metadata.loc[metadata["class_name"] == RirLocation.OUTDOOR.value]["filename"])]


###############################################################################
# ! EXPORTS
###############################################################################
def make_location_rir(path_to_dataset: str,
                      rir_location: RirLocation,
                      p: float = 0.5,
                      *,
                      cahce_size: int = 128,
                      ):
    fq_metadata_file = os.path.join(path_to_dataset, "metadata.csv")
    metadata = pd.read_csv(fq_metadata_file)
    filtered = [os.path.join(path_to_dataset, "data", f) for f in _filter(metadata, rir_location)]
    return ApplyImpulseResponse(filtered, p=p, lru_cache_size=cahce_size)
