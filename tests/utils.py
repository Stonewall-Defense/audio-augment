###############################################################################
# Global Imports
###############################################################################
import os

###############################################################################
# Global Imports
###############################################################################
import numpy as np
from numpy.typing import NDArray
import scipy.signal
import torch


###############################################################################
# Constants
###############################################################################
TEST_DIR = os.path.dirname(__file__)
TEST_RES_DIR = os.path.join(TEST_DIR, "res")


###############################################################################
# Functions
###############################################################################
def fast_autocorr(original: NDArray | torch.Tensor, delayed: NDArray | torch.Tensor, t: int = 0):
    """Only every 4th sample is considered in order to improve execution time"""

    if isinstance(original, torch.Tensor):
        original = original.numpy()

    if isinstance(delayed, torch.Tensor):
        delayed = delayed.numpy()

    if t == 0:
        return np.corrcoef([original[::4], delayed[::4]])[1, 0]
    elif t < 0:
        return np.corrcoef([original[-t::4], delayed[:t:4]])[1, 0]
    else:
        return np.corrcoef([original[:-t:4], delayed[t::4]])[1, 0]


def get_chirp_test(sample_rate: int, duration: float):
    """Create a `duration` seconds chirp from 0Hz to `Nyquist frequency`"""
    n = np.arange(0, duration, 1 / sample_rate)
    samples = scipy.signal.chirp(n, 0, duration, sample_rate // 2, method="linear")
    return torch.from_numpy(samples)


def get_randn_test(sample_rate: int, duration: float):
    """Create a random noise test stimulus"""
    n_samples = int(duration * sample_rate)
    samples = np.random.randn(n_samples)
    return torch.from_numpy(samples)


def find_best_alignment_offset_with_corr_coef(
    reference_signal: torch.Tensor,
    delayed_signal: torch.Tensor,
    min_offset_samples: int,
    max_offset_samples: int,
    lookahead_samples: int | None = None,
    consider_both_polarities: bool = True,
):
    """
    Returns the estimated delay (in samples) between the original and delayed signal,
    calculated using correlation coefficients. The delay is optimized to maximize the
    correlation between the signals.

    Args:
        reference_signal (NDArray[np.float32]): The original signal array.
        delayed_signal (NDArray[np.float32]): The delayed signal array.
        min_offset_samples (int): The minimum delay offset to consider, in samples.
                                  Can be negative.
        max_offset_samples (int): The maximum delay offset to consider, in samples.
        lookahead_samples (Optional[int]): The number of samples to look at
                                           while estimating the delay. If None, the
                                           whole delayed signal is considered.
        consider_both_polarities (bool): If True, the function will consider both positive
                                         and negative correlations, which corresponds to
                                         the same or opposite polarities in signals,
                                         respectively. Defaults to True.

    Returns:
        tuple: Estimated delay (int) and correlation coefficient (float).
    """
    _reference_signal = reference_signal.numpy()
    _delayed_signal = delayed_signal.numpy()

    if lookahead_samples is not None and len(_reference_signal) > lookahead_samples:
        middle_of_signal_index = int(np.floor(len(_reference_signal) / 2))
        original_signal_slice = _reference_signal[
            middle_of_signal_index:middle_of_signal_index + lookahead_samples
        ]
        delayed_signal_slice = _delayed_signal[
            middle_of_signal_index:middle_of_signal_index + lookahead_samples
        ]
    else:
        original_signal_slice = _reference_signal
        delayed_signal_slice = _delayed_signal

    coefs = []
    for lag in range(min_offset_samples, max_offset_samples):
        correlation_coef = fast_autocorr(
            original_signal_slice, delayed_signal_slice, t=lag
        )
        coefs.append(correlation_coef)

    if consider_both_polarities:
        # In this mode we aim to find the correlation coefficient of highest magnitude.
        # We do this to consider the possibility that the delayed signal has opposite
        # polarity compared to the original signal, in which case the correlation
        # coefficient would be negative.
        most_extreme_coef_index = int(np.argmax(np.abs(coefs)))
    else:
        most_extreme_coef_index = int(np.argmax(coefs))
    most_extreme_coef = coefs[most_extreme_coef_index]
    offset = most_extreme_coef_index + min_offset_samples
    return offset, most_extreme_coef
