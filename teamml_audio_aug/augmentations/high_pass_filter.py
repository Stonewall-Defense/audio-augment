###############################################################################
# Global Imports
###############################################################################
import random
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfilt_zi
import torch

###############################################################################
# Certus Imports
###############################################################################
from AudioMlSpecTools import mel_to_hz, hz_to_mel, MelType

###############################################################################
# Local Imports
###############################################################################
from teamml_audio_aug.core.transforms_interface import BaseWaveformTransform


###############################################################################
# Exports
###############################################################################
class HighPassFilter(BaseWaveformTransform):
    """
    Apply high-pass filtering to the input audio of parametrized filter steepness (6/12/18... dB / octave).
    Can also be set for zero-phase filtering (will result in a 6 dB drop at cutoff).
    """

    supports_multichannel = True

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        min_cutoff_freq: float = 20.0,
        max_cutoff_freq: float = 2400.0,
        min_rolloff: int = 12,
        max_rolloff: int = 24,
        zero_phase: bool = False,
        p: float = 0.5,
    ):
        """
        :param min_cutoff_freq: Minimum cutoff frequency in hertz
        :param max_cutoff_freq: Maximum cutoff frequency in hertz
        :param min_rolloff: Minimum filter roll-off (in dB/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in dB/octave)
            Must be a multiple of 6
        :param zero_phase: Whether filtering should be zero phase.
            When this is set to `True`, it will not affect the phase of the
            input signal but will sound 3 dB lower at the cutoff frequency
            compared to the non-zero phase case (6 dB vs. 3 dB). Additionally,
            it is twice as slow as the non-zero phase case. If you
            absolutely want no phase distortions (e.g. want to augment a
            drum track), set this to `True`.
        :param p: The probability of applying this transform
        """
        super().__init__(sample_rate, p=p)

        if min_cutoff_freq <= 0:
            raise ValueError(
                f"HighPassFilter requires min_cutoff_freq > 0. Got {min_cutoff_freq}."
            )

        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        self.min_rolloff = min_rolloff
        self.max_rolloff = max_rolloff
        self.zero_phase = zero_phase

        if self.zero_phase:
            assert (
                self.min_rolloff % 12 == 0
            ), "Zero phase filters can only have a steepness which is a multiple of 12 dB/octave"
            assert (
                self.max_rolloff % 12 == 0
            ), "Zero phase filters can only have a steepness which is a multiple of 12 dB/octave"
        else:
            assert (
                self.min_rolloff % 6 == 0
            ), "Non zero phase filters can only have a steepness which is a multiple of 6 dB/octave"
            assert (
                self.max_rolloff % 6 == 0
            ), "Non zero phase filters can only have a steepness which is a multiple of 6 dB/octave"

        if self.min_cutoff_freq > self.max_cutoff_freq:
            raise ValueError("min_cutoff_freq must not be greater than max_cutoff_freq")
        if self.min_rolloff > self.max_rolloff:
            raise ValueError("min_rolloff must not be greater than max_rolloff")

        self.rolloff = self.min_rolloff
        self.cutoff_freq = self.min_cutoff_freq

    def randomize_parameters(self, samples: torch.Tensor):
        super().randomize_parameters(samples)
        if self.zero_phase:
            random_order = random.randint(
                self.min_rolloff // 12, self.max_rolloff // 12
            )
            self.rolloff = random_order * 12
        else:
            random_order = random.randint(self.min_rolloff // 6, self.max_rolloff // 6)
            self.rolloff = random_order * 6

        cutoff_mel = np.random.uniform(
            low=hz_to_mel(self.min_cutoff_freq, MelType.OSHAUGHNESSY),
            high=hz_to_mel(self.max_cutoff_freq, MelType.OSHAUGHNESSY),
        )
        self.cutoff_freq = mel_to_hz(cutoff_mel, MelType.OSHAUGHNESSY)

    def apply(self, samples: torch.Tensor) -> torch.Tensor:
        cutoff_freq = self.cutoff_freq
        nyquist_freq = self.sample_rate // 2
        if cutoff_freq > nyquist_freq:
            # Ensure that the cutoff frequency does not exceed the Nyquist
            # frequency to avoid an exception from scipy
            cutoff_freq = nyquist_freq * 0.9999

        sos = butter(
            self.rolloff // (12 if self.zero_phase else 6),
            cutoff_freq,
            btype="highpass",
            analog=False,
            fs=self.sample_rate,
            output="sos",
        )

        # The actual processing takes place here
        samples_tmp = samples.numpy()

        if len(samples_tmp.shape) == 1:
            if self.zero_phase:
                processed_samples = sosfiltfilt(sos, samples_tmp)
            else:
                processed_samples, _ = sosfilt(
                    sos, samples_tmp, zi=sosfilt_zi(sos) * samples_tmp[0]
                )
            processed_samples = processed_samples.astype(np.float32)
        else:
            processed_samples = np.zeros_like(samples_tmp, dtype=np.float32)
            if self.zero_phase:
                for chn_idx in range(samples_tmp.shape[0]):
                    processed_samples[chn_idx, :] = sosfiltfilt(
                        sos, samples_tmp[chn_idx, :]
                    )
            else:
                zi = sosfilt_zi(sos)
                for chn_idx in range(samples_tmp.shape[0]):
                    processed_samples[chn_idx, :], _ = sosfilt(
                        sos, samples_tmp[chn_idx, :], zi=zi * samples_tmp[chn_idx, 0]
                    )

        return torch.from_numpy(processed_samples)
