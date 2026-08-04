###############################################################################
# Global Imports
###############################################################################
import logging
import math
import random
from typing import Literal, Optional

###############################################################################
# 3PP Imports
###############################################################################
import numpy as np
import numpy_audio_limiter

###############################################################################
# Local Imports
###############################################################################
from teamml_audio_augment.core.transforms_interface import BaseWaveformTransform
from teamml_audio_augment.core.utils import (
    convert_decibels_to_amplitude_ratio,
    get_max_abs_amplitude,
)


###############################################################################
# Config
###############################################################################
LOGGER = logging.getLogger("teamMl")


###############################################################################
# Exports
###############################################################################
class Limiter(BaseWaveformTransform):
    """
    A simple audio limiter (dynamic range compression).
    """

    supports_multichannel = True

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        min_threshold_db: float = -24.0,
        max_threshold_db: float = -2.0,
        min_attack: float = 0.0005,
        max_attack: float = 0.025,
        min_release: float = 0.05,
        max_release: float = 0.7,
        threshold_mode: Literal[
            "relative_to_signal_peak", "absolute"
        ] = "relative_to_signal_peak",
        p: float = 0.5,
    ):
        """
        The threshold determines the audio level above which the limiter kicks in.
        The attack time is how quickly the limiter kicks in once the audio signal starts exceeding the threshold.
        The release time determines how quickly the limiter stops working after the signal drops below the threshold.

        :param min_threshold_db: Minimum threshold in Decibels
        :param max_threshold_db: Maximum threshold in Decibels
        :param min_attack: Minimum attack time in seconds
        :param max_attack: Maximum attack time in seconds
        :param min_release: Minimum release time in seconds
        :param max_release: Maximum release time in seconds
        :param threshold_mode: "relative_to_signal_peak" or "absolute"
            "relative_to_signal_peak" means the threshold is relative to peak of the signal.
            "absolute" means the threshold is relative to 0 dBFS, so it doesn't depend
            on the peak of the signal.
        :param p: The probability of applying this transform
        """
        super().__init__(sample_rate, p=p)

        assert min_attack > 0.0, "min_attack must be a positive number"
        assert max_attack > 0.0, "max_attack must be a positive number"
        assert (
            max_attack >= min_attack
        ), "max_attack must be greater than or equal to min_attack"
        assert min_release > 0.0, "min_release must be a positive number"
        assert max_release > 0.0, "max_release must be a positive number"
        assert (
            max_release >= min_release
        ), "max_release must be greater than or equal to min_release"
        assert (
            max_threshold_db >= min_threshold_db
        ), "max_threshold_db must be greater than or equal to min_threshold_db"
        assert threshold_mode in (
            "relative_to_signal_peak",
            "absolute",
        ), 'threshold_mode must be either "relative_to_signal_peak" or "absolute"'
        self.min_threshold_db = min_threshold_db
        self.max_threshold_db = max_threshold_db
        self.min_attack = min_attack
        self.max_attack = max_attack
        self.min_release = min_release
        self.max_release = max_release
        self.threshold_mode = threshold_mode

        self.randomize_parameters(np.zeros(1))

    @staticmethod
    def convert_time_to_coefficient(
        t: float, sample_rate: int, decay_threshold: Optional[float] = None
    ) -> float:
        if decay_threshold is None:
            # Attack time and release time in this transform are defined as how long
            # it takes to step 1-decay_threshold of the way to a constant target gain.
            # The default threshold used here is inspired by RT60.
            decay_threshold = convert_decibels_to_amplitude_ratio(-60)
        return 10 ** (math.log10(decay_threshold) / max(sample_rate * t, 1.0))

    def randomize_parameters(self, samples: np.ndarray):
        super().randomize_parameters(samples)

        if self.should_apply:
            attack_seconds = random.uniform(self.min_attack, self.max_attack)
            self.attack = self.convert_time_to_coefficient(
                attack_seconds, self.sample_rate
            )
            release_seconds = random.uniform(self.min_release, self.max_release)
            self.release = self.convert_time_to_coefficient(
                release_seconds, self.sample_rate
            )
            # Delay the signal by 60% of the attack time by default
            self.delay = max(round(0.6 * attack_seconds * self.sample_rate), 1)

            threshold_factor = (
                get_max_abs_amplitude(samples)
                if self.threshold_mode == "relative_to_signal_peak"
                else 1.0
            )
            threshold_db = random.uniform(self.min_threshold_db, self.max_threshold_db)

            self.threshold = float(
                threshold_factor * convert_decibels_to_amplitude_ratio(threshold_db)
            )

    def apply(self, samples: np.ndarray) -> np.ndarray:
        if self.threshold == 0.0:
            # Digital silence input can cause this to happen
            return samples

        original_ndim = samples.ndim
        if original_ndim == 1:
            samples = samples.reshape((1, -1))
        elif samples.shape[0] > 1 and not samples.flags.c_contiguous:
            samples = np.ascontiguousarray(samples)

        processed_samples = numpy_audio_limiter.limit(
            signal=samples.astype(np.float32),
            attack_coeff=self.attack,
            release_coeff=self.release,
            delay=self.delay,
            threshold=self.threshold,
        )
        if original_ndim == 1:
            processed_samples = processed_samples[0]

        LOGGER.debug(f"Applied limiter of {self.attack:.03f}/{self.release:.03f}/{self.delay:.03f}/{self.threshold:.03f}")

        return processed_samples
