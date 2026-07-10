###############################################################################
# Global Imports
###############################################################################
import random
from typing import Literal, Optional
import warnings

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
    convert_decibels_to_amplitude_ratio,
)


###############################################################################
# Helpers
###############################################################################
def _oscillator_bank(
    frequencies: torch.Tensor,
    amplitudes: torch.Tensor,
    sample_rate: float,
    reduction: str = "sum",
    dtype: Optional[torch.dtype] = torch.float64,
) -> torch.Tensor:
    """Synthesize waveform from the given instantaneous frequencies and amplitudes.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Note:
        The phase information of the output waveform is found by taking the cumulative sum
        of the given instantaneous frequencies (``frequencies``).
        This incurs roundoff error when the data type does not have enough precision.
        Using ``torch.float64`` can work around this.

        The following figure shows the difference between ``torch.float32`` and
        ``torch.float64`` when generating a sin wave of constant frequency and amplitude
        with sample rate 8000 [Hz].
        Notice that ``torch.float32`` version shows artifacts that are not seen in
        ``torch.float64`` version.

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/oscillator_precision.png

    Args:
        frequencies (Tensor): Sample-wise oscillator frequencies (Hz). Shape `(..., time, N)`.
        amplitudes (Tensor): Sample-wise oscillator amplitude. Shape: `(..., time, N)`.
        sample_rate (float): Sample rate
        reduction (str): Reduction to perform.
            Valid values are ``"sum"``, ``"mean"`` or ``"none"``. Default: ``"sum"``
        dtype (torch.dtype or None, optional): The data type on which cumulative sum operation is performed.
            Default: ``torch.float64``. Pass ``None`` to disable the casting.

    Returns:
        Tensor:
            The resulting waveform.

            If ``reduction`` is ``"none"``, then the shape is
            `(..., time, N)`, otherwise the shape is `(..., time)`.
    """
    if frequencies.shape != amplitudes.shape:
        raise ValueError(
            "The shapes of `frequencies` and `amplitudes` must match. "
            f"Found: {frequencies.shape} and {amplitudes.shape} respectively."
        )
    reductions = ["sum", "mean", "none"]
    if reduction not in reductions:
        raise ValueError(f"The value of reduction must be either {reductions}. Found: {reduction}")

    invalid = torch.abs(frequencies) >= sample_rate / 2
    if torch.any(invalid):
        warnings.warn(
            "Some frequencies are above nyquist frequency. "
            "Setting the corresponding amplitude to zero. "
            "This might cause numerically unstable gradient."
        )
        amplitudes = torch.where(invalid, 0.0, amplitudes)

    pi2 = 2.0 * torch.pi
    freqs = frequencies * pi2 / sample_rate % pi2
    phases = torch.cumsum(freqs, dim=-2, dtype=dtype)
    if dtype is not None and freqs.dtype != dtype:
        phases = phases.to(freqs.dtype)

    waveform = amplitudes * torch.sin(phases)
    if reduction == "sum":
        return waveform.sum(-1)
    if reduction == "mean":
        return waveform.mean(-1)
    return waveform


###############################################################################
# Classes
###############################################################################
class Infrasound(BaseWaveformTransform):
    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        min_freq_hz=2,
        max_freq_hz=50,
        min_infra_freqs=2,
        max_infra_freqs=8,
        noise_level_type: Literal["absolute", "relative"] = "relative",
        min_snr_db=3.0,
        max_snr_db=30.0,
        min_absolute_rms_db=-45.0,
        max_absolute_rms_db=-15.0,
        p=0.5,
    ):
        super().__init__(sample_rate, p=p)

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

        self.freqs: list[torch.Tensor] = []
        self.amps: list[float] = []

    def randomize_parameters(self, samples: torch.Tensor):
        super().randomize_parameters(samples)
        if self.should_apply:
            num_freqs = random.randint(self.min_infra_freqs, self.max_infra_freqs)
            choices = random.sample(self.freq_range_hz, num_freqs)
            self.freqs = [self._make_freq(samples.shape[-1], F0, self.sample_rate) for F0 in choices]
            self.amps = [self._make_amp() for _ in range(len(choices))]

    def apply(self, samples: torch.Tensor) -> torch.Tensor:
        clean_rms = calculate_rms(samples)

        for noise, amp in zip(self.freqs, self.amps):
            noise_rms = calculate_rms(noise)

            if self.noise_level_type == "relative":
                desired_noise_rms = calculate_desired_noise_rms(clean_rms, amp)
                gain = desired_noise_rms / noise_rms
            else:
                desired_noise_rms_amp = convert_decibels_to_amplitude_ratio(amp)
                gain = desired_noise_rms_amp / noise_rms

            noise = noise * gain
            samples += noise

        return samples

    def _make_freq(self, num_samples: int, F0: int, sr: int) -> torch.Tensor:
        freq = torch.full((num_samples, 1), F0)
        amp = torch.ones((num_samples, 1))

        # Phase shift to avoid artifacts at start and end of range
        waveform = _oscillator_bank(freq, amp, sample_rate=sr)
        shift_amount = random.uniform(0, 1)
        num_places_to_shift = int(round(shift_amount * num_samples))
        shifted_wave = torch.roll(waveform, num_places_to_shift, dims=-1)

        return shifted_wave

    def _make_amp(self):
        if self.noise_level_type == "absolute":
            return random.uniform(self.min_absolute_rms_db, self.max_absolute_rms_db)
        else:
            return random.uniform(self.min_snr_db, self.max_snr_db)
