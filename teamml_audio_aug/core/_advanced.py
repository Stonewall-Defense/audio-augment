###############################################################################
# Global Imports
###############################################################################
import ctypes
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


###############################################################################
# Exports
###############################################################################
def rms(a: torch.Tensor, window_size: Optional[int] = None) -> torch.Tensor:
    """
    Calculate RMS series for the given NumPy array.

    :param a: NumPy array to process. Can be 1D or 2D.
    :param window_size: Window size for the RMS calculation. If not specified, it defaults to the length of the array.
    :return: A NumPy array containing the RMS series.
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
                _numpy_rms.ffi.cast("float *", ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_float))),
                window_size,
                _numpy_rms.ffi.cast("float *", ctypes.cast(output_array.data_ptr(), ctypes.POINTER(ctypes.c_float))),
                output_length,
            )
        else:  # a.ndim == 2
            for i in range(a.shape[0]):
                _numpy_rms.lib.rms(
                    _numpy_rms.ffi.cast("float *", ctypes.cast(a[i].data_ptr(), ctypes.POINTER(ctypes.c_float))),
                    window_size,
                    _numpy_rms.ffi.cast("float *", ctypes.cast(output_array[i].data_ptr(), ctypes.POINTER(ctypes.c_float))),
                    output_length,
                )

        return output_array

    return _rms_fallback(a, window_size)


def minmax(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if 0 in a.shape:
        raise ValueError("Cannot find min/max value in empty array")

    if a.dtype == torch.float32 and a.is_contiguous():
        result = _numpy_minmax.lib.minmax_contiguous_float32(
            _numpy_minmax.ffi.cast("float *", ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_float))), a.size
        )
        return torch.tensor(result.min_val, dtype=torch.float32), torch.tensor(result.max_val, dtype=torch.float32)

    return torch.amin(a), torch.amax(a)
