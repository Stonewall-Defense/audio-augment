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
class EscClass(Enum):
    ANIMALS = 0
    NATURAL_SOUNDSCAPES = 10
    HUMAN_NONSPEECH = 20
    INTERIOR_DOMESTIC = 30
    EXTERIOR_URBAN = 40


###############################################################################
# Constants
###############################################################################
ESC_REGEX = re.compile(r'^\d+-\w+-[A-Z]-(?P<class_id>\d+)\.wav$')


###############################################################################
# Helpers
###############################################################################
def _file_matches_classes(filename: str, classes: list[EscClass]) -> bool:
    matches = re.match(ESC_REGEX, filename)
    if not matches:
        return False

    class_id = int(matches.group("class_id"))

    for c in classes:
        min_val = c.value
        max_val = min_val + 9
        if min_val <= class_id <= max_val:
            return True

    return False


###############################################################################
# ! EXPORTS
###############################################################################
def create_esc_augment(path_to_esc50: str,
                       classes: list[EscClass],
                       min_snr_db: float = 3.0,
                       max_snr_db: float = 30.0,
                       p: float = 0.5,
                       max_samples: int = 1_000,
                       ) -> AddBackgroundNoise:
    classes = classes or [val for val in EscClass]

    audio_files = [os.path.join(path_to_esc50, f) for f in os.listdir(path_to_esc50) if f.endswith(".wav") if _file_matches_classes(f, classes)]
    random.shuffle(audio_files)
    audio_files = audio_files[:max_samples]

    return AddBackgroundNoise(audio_files, min_snr_db=min_snr_db, max_snr_db=max_snr_db, p=p)
