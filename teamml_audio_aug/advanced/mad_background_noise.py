###############################################################################
# Global Imports
###############################################################################
from enum import Enum
import os
import random
import re

###############################################################################
# 3PP Imports
###############################################################################
from audiomentations import AddBackgroundNoise


###############################################################################
# Enums
###############################################################################
class MadClass(Enum):
    COMMUNICATION = "communications"
    GUNSHOT = "gunshot"
    FOOTSTEPS = "footsteps"
    SHELLING = "shelling"
    VEHICLE = "vehicle"
    HELICOPTER = "helicopter"
    FIGHTER = "fighter"


###############################################################################
# Helpers
###############################################################################
def _make_mad_bg_common(mad_files: list[str],
                        min_snr_db: float = 3.0,
                        max_snr_db: float = 30.0,
                        p: float = 0.5,
                        max_samples: int = 10_000,
                        ):
    random.shuffle(mad_files)
    mad_files = mad_files[:max_samples]

    return AddBackgroundNoise(mad_files, min_snr_db=min_snr_db, max_snr_db=max_snr_db, p=p)


###############################################################################
# ! EXPORTS
###############################################################################
def make_mad_bg_single_class(path_to_mad,
                             mad_class: MadClass,
                             min_snr_db: float = 3.0,
                             max_snr_db: float = 30.0,
                             p: float = 0.5,
                             max_samples: int = 10_000,
                             ):
    mad_files = [os.path.join(path_to_mad, f) for f in os.listdir(path_to_mad) if f.endswith(".wav") and mad_class.value in f]
    return _make_mad_bg_common(mad_files, min_snr_db, max_snr_db, p, max_samples)


def make_mad_bg_exclude(path_to_mad: str,
                        class_to_exclude: MadClass,
                        min_snr_db: float = 3.0,
                        max_snr_db: float = 30.0,
                        p: float = 0.5,
                        max_samples: int = 10_000,
                        ):
    mad_files = [os.path.join(path_to_mad, f) for f in os.listdir(path_to_mad) if f.endswith(".wav") and class_to_exclude.value not in f]
    return _make_mad_bg_common(mad_files, min_snr_db, max_snr_db, p, max_samples)


def make_mad_bg_multi_class(path_to_mad: str,
                            mad_classes: list[MadClass],
                            min_snr_db: float = 3.0,
                            max_snr_db: float = 30.0,
                            p: float = 0.5,
                            max_samples: int = 10_000,
                            ):
    pattern = re.compile(f"(?:{'|'.join(c.value[0] for c in mad_classes)}).+.wav")
    mad_files = [os.path.join(path_to_mad, f) for f in os.listdir(path_to_mad) if re.search(pattern, f)]
    return _make_mad_bg_common(mad_files, min_snr_db, max_snr_db, p, max_samples)
