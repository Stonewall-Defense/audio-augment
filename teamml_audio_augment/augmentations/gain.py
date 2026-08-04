###############################################################################
# Global Imports
###############################################################################
import logging
import random
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import numpy as np

###############################################################################
# Local Imports
###############################################################################
from teamml_audio_augment.core.transforms_interface import BaseWaveformTransform
from teamml_audio_augment.core.utils import convert_decibels_to_amplitude_ratio


###############################################################################
# Config
###############################################################################
LOGGER = logging.getLogger("teamMl")


###############################################################################
# Exports
###############################################################################
class Gain(BaseWaveformTransform):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.

    Warning: This transform can return samples outside the [-1, 1] range, which may lead to
    clipping or wrap distortion, depending on what you do with the audio in a later stage.
    See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping
    """

    supports_multichannel = True

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        p: float = 0.5,
    ):
        """
        :param min_gain_db: Minimum gain
        :param max_gain_db: Maximum gain
        :param p: The probability of applying this transform
        """
        super().__init__(sample_rate, p=p)

        if min_gain_db > max_gain_db:
            raise ValueError("min_gain_db cannot be greater than max_gain_db")

        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        self.amplitude_ratio = convert_decibels_to_amplitude_ratio(self.min_gain_db)

    def randomize_parameters(self, samples: np.ndarray):
        super().randomize_parameters(samples)
        if self.should_apply:
            self.amplitude_ratio = convert_decibels_to_amplitude_ratio(
                random.uniform(self.min_gain_db, self.max_gain_db)
            )

    def apply(self, samples: np.ndarray) -> np.ndarray:
        LOGGER.debug(f"Appied gain @ {self.amplitude_ratio:.03f}")
        return samples * self.amplitude_ratio
