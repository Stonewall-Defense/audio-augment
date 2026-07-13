###############################################################################
# Global Imports
###############################################################################
from enum import Enum
import os
from pathlib import Path
import random
import re
from typing import Literal, Optional

###############################################################################
# Local Imports
###############################################################################
from teamml_audio_augment.augmentations.add_background_noise import AddBackgroundNoise


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


MAD_CLASS = Literal[
    "communications",
    "gunshot",
    "footsteps",
    "shelling",
    "vehicle",
    "helicopter",
    "fighter",
]


###############################################################################
# Helpers
###############################################################################
def _make_mad_bg_common(mad_files: list[str],
                        sample_rate: Optional[int],
                        min_snr_db: float,
                        max_snr_db: float,
                        p: float,
                        max_samples: int,
                        ):
    random.shuffle(mad_files)
    mad_files = mad_files[:max_samples]

    return AddBackgroundNoise(mad_files, sample_rate=sample_rate, min_snr_db=min_snr_db, max_snr_db=max_snr_db, p=p)


###############################################################################
# ! EXPORTS
###############################################################################
def make_mad_bg_single_class(path_to_mad: str | Path,
                             mad_class: MadClass | MAD_CLASS,
                             *,
                             sample_rate: Optional[int] = None,
                             min_snr_db: float = 3.0,
                             max_snr_db: float = 30.0,
                             p: float = 0.5,
                             max_samples: int = 10_000,
                             ):
    mc = mad_class.value if isinstance(mad_class, MadClass) else mad_class

    mad_files = [os.path.join(path_to_mad, f) for f in os.listdir(path_to_mad) if f.endswith(".wav") and mc in f]
    return _make_mad_bg_common(mad_files, sample_rate, min_snr_db, max_snr_db, p, max_samples)


def make_mad_bg_exclude(path_to_mad: str | Path,
                        class_to_exclude: MadClass | MAD_CLASS,
                        *,
                        sample_rate: Optional[int] = None,
                        min_snr_db: float = 3.0,
                        max_snr_db: float = 30.0,
                        p: float = 0.5,
                        max_samples: int = 10_000,
                        ):
    cte = class_to_exclude.value if isinstance(class_to_exclude, MadClass) else class_to_exclude

    mad_files = [os.path.join(path_to_mad, f) for f in os.listdir(path_to_mad) if f.endswith(".wav") and cte not in f]
    return _make_mad_bg_common(mad_files, sample_rate, min_snr_db, max_snr_db, p, max_samples)


def make_mad_bg_multi_class(path_to_mad: str | Path,
                            mad_classes: list[MadClass] | list[MAD_CLASS],
                            *,
                            sample_rate: Optional[int] = None,
                            min_snr_db: float = 3.0,
                            max_snr_db: float = 30.0,
                            p: float = 0.5,
                            max_samples: int = 10_000,
                            ):
    mc = [mad.value if isinstance(mad, MadClass) else mad for mad in mad_classes]

    pattern = re.compile(f"(?:{'|'.join(c[:3] for c in mc)}).+.wav")
    mad_files = [os.path.join(path_to_mad, f) for f in os.listdir(path_to_mad) if re.search(pattern, f)]
    return _make_mad_bg_common(mad_files, sample_rate, min_snr_db, max_snr_db, p, max_samples)
