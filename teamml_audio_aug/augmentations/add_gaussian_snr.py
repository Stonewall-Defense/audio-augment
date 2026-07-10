###############################################################################
# Global Imports
###############################################################################
import random
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import torch

###############################################################################
# Local Imports
###############################################################################
from teamml_audio_aug.core.transforms_interface import BaseWaveformTransform
from teamml_audio_aug.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
)


###############################################################################
# Exports
###############################################################################
class AddGaussianSNR(BaseWaveformTransform):
    """
    Add gaussian noise to the input. A random Signal to Noise Ratio (SNR) will be picked
    uniformly in the Decibel scale. This aligns with human hearing, which is more
    logarithmic than linear.
    """

    supports_multichannel = True

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        min_snr_db: float = 5.0,
        max_snr_db: float = 40.0,
        p: float = 0.5,
    ):
        """
        :param min_snr_db: Minimum signal-to-noise ratio in dB. A lower number means more noise.
        :param max_snr_db: Maximum signal-to-noise ratio in dB. A greater number means less noise.
        :param p: The probability of applying this transform
        """
        super().__init__(sample_rate, p=p)

        if min_snr_db > max_snr_db:
            raise ValueError("min_snr_db must not be greater than max_snr_db")
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        self.noise_std = 0.0

    def randomize_parameters(self, samples: torch.Tensor):
        super().randomize_parameters(samples)
        if self.should_apply:
            # Pick SNR in decibel scale
            snr = random.uniform(self.min_snr_db, self.max_snr_db)

            clean_rms = calculate_rms(samples)
            noise_rms = calculate_desired_noise_rms(clean_rms=clean_rms, snr=snr)

            # In gaussian noise, the RMS gets roughly equal to the std
            self.noise_std = float(noise_rms)

    def apply(self, samples: torch.Tensor) -> torch.Tensor:
        noise = torch.normal(0.0, self.noise_std, size=samples.shape)
        return samples + noise
