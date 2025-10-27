###############################################################################
# Global Imports
###############################################################################
from enum import Enum
import random

###############################################################################
# 3PP Imports
###############################################################################
import numpy as np
from numpy.typing import NDArray
import torch
from torchaudio.prototype.functional import oscillator_bank

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
    convert_decibels_to_amplitude_ratio,
)


###############################################################################
# Enums
###############################################################################
class NoiseLevelType(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


###############################################################################
# Classes
###############################################################################
class Infrasound(BaseWaveformTransform):
    def __init__(
        self,
        min_freq_hz=2,
        max_freq_hz=50,
        min_infra_freqs=2,
        max_infra_freqs=8,
        *,
        noise_level_type: NoiseLevelType = NoiseLevelType.RELATIVE,
        min_snr_db=3.0,
        max_snr_db=30.0,
        min_absolute_rms_db=-45.0,
        max_absolute_rms_db=-15.0,
        p=0.5,
    ):
        super().__init__(p)

        if min_freq_hz > max_freq_hz:
            raise ValueError("min_freq_hz must not be greater than max_freq_hz")
        elif min_infra_freqs > max_infra_freqs:
            raise ValueError("min_infra_freqs must not be greater than max_infra_freqs")
        elif min_snr_db > max_snr_db:
            raise ValueError("min_snr_db must not be greater than max_snr_db")
        elif min_absolute_rms_db > max_absolute_rms_db:
            raise ValueError("min_absolute_rms_db must not be greater than max_absolute_rms_db")

        self.freq_range_hz = [hz for hz in range(min_freq_hz, max_freq_hz + 1)]
        self.min_infra_freqs = min_infra_freqs
        self.max_infra_freqs = max_infra_freqs
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.min_absolute_rms_db = min_absolute_rms_db
        self.max_absolute_rms_db = max_absolute_rms_db
        self.noise_level_type = noise_level_type

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            num_freqs = random.randint(self.min_infra_freqs, self.max_infra_freqs)
            choices = random.sample(self.freq_range_hz, num_freqs)
            self.parameters["freqs"] = [self._make_freq(samples.size, F0, sample_rate) for F0 in choices]
            self.parameters["amps"] = [self._make_amp() for _ in range(len(choices))]

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        clean_rms = calculate_rms(samples)

        for noise, amp in zip(self.parameters["freqs"], self.parameters["amps"]):
            noise_rms = calculate_rms(noise)

            if self.noise_level_type == NoiseLevelType.RELATIVE:
                desired_noise_rms = calculate_desired_noise_rms(clean_rms, amp)
                gain = desired_noise_rms / noise_rms
            else:
                desired_noise_rms_amp = convert_decibels_to_amplitude_ratio(amp)
                gain = desired_noise_rms_amp / noise_rms

            noise = noise * gain
            samples += noise

        return samples

    def _make_freq(self, num_samples: int, F0: int, sr: int) -> NDArray[np.float32]:
        freq = torch.full((num_samples, 1), F0)
        amp = torch.ones((num_samples, 1))

        # Phase shift to avoid artifacts at start and end of range
        waveform = oscillator_bank(freq, amp, sample_rate=sr).numpy()
        shift_amount = random.uniform(0, 1)
        num_places_to_shift = int(round(shift_amount * num_samples))
        shifted_wave = np.roll(waveform, num_places_to_shift, axis=-1)

        return shifted_wave

    def _make_amp(self):
        if self.noise_level_type == NoiseLevelType.ABSOLUTE:
            return random.uniform(self.min_absolute_rms_db, self.max_absolute_rms_db)
        else:
            return random.uniform(self.min_snr_db, self.max_snr_db)
