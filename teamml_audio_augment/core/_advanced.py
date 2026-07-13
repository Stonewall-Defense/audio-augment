###############################################################################
# Global Imports
###############################################################################
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
from numpy_minmax import _numpy_minmax
from numpy_rms import _numpy_rms

import torch


###############################################################################
# Falback Helpers
###############################################################################
def _calculate_rms(a: torch.Tensor):
    """Given a numpy array, return its RMS power level."""
    return torch.sqrt(torch.mean(torch.square(a), dim=-1))


def _rms_fallback(a: torch.Tensor, window_size: int) -> torch.Tensor:
    if 0 in a.shape:
        raise ValueError("Cannot input empty array")

    output_shape = a.shape[:-1] + (a.shape[-1] // window_size,)
    output_array = torch.zeros(output_shape, dtype=a.dtype)

    end_index = output_shape[-1] * window_size

    output_i = 0
    for offset in range(0, end_index, window_size):
        rms = _calculate_rms(a[..., offset:offset + window_size])
        output_array[..., output_i] = rms
        output_i += 1
    return output_array


def _next_fast_len(n: int) -> int:
    """Find the next integer with only small prime factors (2, 3, 5)."""
    best = n
    p2 = 1
    while p2 < n:
        p3 = p2
        while p3 < n:
            p5 = p3
            while p5 < n:
                p5 *= 5
            if p5 < best:
                best = p5
            p3 *= 3
        p2 *= 2
    return best


###############################################################################
# Exports
###############################################################################
def rms(a: torch.Tensor, window_size: Optional[int] = None) -> torch.Tensor:
    """
        This is a minimal PyTorch version of the same code from https://pypi.org/project/numpy-rms/
    """

    if 0 in a.shape:
        raise ValueError("Cannot input empty array")

    if window_size is None:
        window_size = a.shape[-1]

    if (a.dtype == torch.float32 and a.ndim in (1, 2) and a.is_contiguous()):
        output_length = a.shape[-1] // window_size
        if a.ndim == 1:
            output_shape = (output_length,)
        else:  # a.ndim == 2
            output_shape = (a.shape[0], output_length)
        output_array = torch.zeros(output_shape, dtype=a.dtype)

        if a.ndim == 1:
            _numpy_rms.lib.rms(
                _numpy_rms.ffi.cast("float *", a.data_ptr()),
                window_size,
                _numpy_rms.ffi.cast("float *", output_array.data_ptr()),
                output_length,
            )
        else:  # a.ndim == 2
            for i in range(a.shape[0]):
                _numpy_rms.lib.rms(
                    _numpy_rms.ffi.cast("float *", a[i].data_ptr()),
                    window_size,
                    _numpy_rms.ffi.cast("float *", output_array[i].data_ptr()),
                    output_length,
                )

        return output_array

    return _rms_fallback(a, window_size)


def minmax(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
        This is a minimal PyTorch version of the same code from https://pypi.org/project/numpy-minmax/
    """

    if 0 in a.shape:
        raise ValueError("Cannot find min/max value in empty array")

    if a.dtype == torch.float32 and a.is_contiguous():
        result = _numpy_minmax.lib.minmax_contiguous_float32(
            _numpy_minmax.ffi.cast("float *", a.data_ptr()), a.size(dim=0)
        )
        return torch.tensor(result.min_val, dtype=torch.float32), torch.tensor(result.max_val, dtype=torch.float32)

    return torch.amin(a), torch.amax(a)


def convolve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.shape[-1] + b.shape[-1] - 1
    fft_len = _next_fast_len(n)
    a_f = torch.fft.rfft(a, n=fft_len)
    b_f = torch.fft.rfft(b, n=fft_len)
    return torch.fft.irfft(a_f * b_f, n=fft_len)[..., :n]
