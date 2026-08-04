###############################################################################
# Global Imports
###############################################################################
from typing import Dict, Optional, Sequence, Tuple, Union
import warnings

###############################################################################
# 3PP Imports
###############################################################################
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.typing import DTypeLike
import scipy


###############################################################################
# Constants
###############################################################################
PAD_MODE = "constant"

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2**8 * 2**10


###############################################################################
# Helpers
###############################################################################
def pad_center(
    data: np.ndarray, *, size: int, axis: int = -1
) -> np.ndarray:
    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(
            f"Target size ({size:d}) must be at least input size ({n:d})"
        )

    return np.pad(data, lengths)


def expand_to(
    x: np.ndarray, *, ndim: int, axes: Union[int, slice, Sequence[int], Sequence[slice]]
) -> np.ndarray:
    axes_tup: Tuple[int]
    try:
        axes_tup = tuple(axes)  # type: ignore
    except TypeError:
        axes_tup = tuple([axes])  # type: ignore

    if len(axes_tup) != x.ndim:
        raise ValueError(
            f"Shape mismatch between axes={axes_tup} and input x.shape={x.shape}"
        )

    if ndim < x.ndim:
        raise ValueError(
            f"Cannot expand x.shape={x.shape} to fewer dimensions ndim={ndim}"
        )

    shape: list[int] = [1] * ndim
    for i, axi in enumerate(axes_tup):
        shape[axi] = x.shape[i]

    return x.reshape(shape)


def frame(
    x: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    axis: int = -1,
    writeable: bool = False,
    subok: bool = False,
) -> np.ndarray:
    # This implementation is derived from numpy.lib.stride_tricks.sliding_window_view (1.20.0)
    # https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
    x = np.array(x, copy=False, subok=subok)

    if x.shape[axis] < frame_length:
        raise ValueError(
            f"Input is too short (n={x.shape[axis]:d}) for frame_length={frame_length:d}"
        )

    if hop_length < 1:
        raise ValueError(f"Invalid hop_length: {hop_length:d}")

    # put our new within-frame axis at the end for now
    out_strides = x.strides + tuple([x.strides[axis]])

    # Reduce the shape on the framing axis
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]


def dtype_r2c(d: DTypeLike, *, default: Optional[type] = np.complex64) -> DTypeLike:
    mapping: Dict[DTypeLike, type] = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
        np.dtype(float): np.dtype(complex).type,
    }

    # If we're given a complex type already, return it
    dt = np.dtype(d)
    if dt.kind == "c":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))


def dtype_c2r(d: DTypeLike, *, default: Optional[type] = np.float32) -> DTypeLike:
    mapping: Dict[DTypeLike, type] = {
        np.dtype(np.complex64): np.float32,
        np.dtype(np.complex128): np.float64,
        np.dtype(complex): np.dtype(float).type,
    }

    # If we're given a real type already, return it
    dt = np.dtype(d)
    if dt.kind == "f":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))


def fix_length(
    data: np.ndarray, *, size: int, axis: int = -1
) -> np.ndarray:
    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths)

    return data


def tiny(x: Union[float, np.ndarray]):
    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.dtype(np.float32)

    return np.finfo(dtype).tiny


def __overlap_add(y, ytmp, hop_length):
    # numba-accelerated overlap add for inverse stft
    # y is the pre-allocated output buffer
    # ytmp is the windowed inverse-stft frames
    # hop_length is the hop-length of the STFT analysis

    n_fft = ytmp.shape[-2]
    N = n_fft
    for frame in range(ytmp.shape[-1]):
        sample = frame * hop_length
        if N > y.shape[-1] - sample:
            N = y.shape[-1] - sample

        y[..., sample:(sample + N)] += ytmp[..., :N, frame]


def __window_ss_fill(x, win_sq, n_frames, hop_length):  # pragma: no cover
    """Compute the sum-square envelope of a window."""
    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]


def normalize(
    S: np.ndarray,
    *,
    norm: Optional[float] = np.inf,
    axis: Optional[int] = 0,
    fill: Optional[bool] = None,
) -> np.ndarray:

    # Avoid div-by-zero
    threshold = tiny(S)

    if fill not in [None, False, True]:
        raise ValueError(f"fill={fill} must be None or boolean")

    if not np.all(np.isfinite(S)):
        raise ValueError("Input must be finite")

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm is None:
        return S

    elif norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ValueError("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    else:
        raise ValueError(f"Unsupported norm: {repr(norm)}")

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def window_sumsquare(
    *,
    window: np.ndarray,
    n_frames: int,
    hop_length: int = 512,
    n_fft: int = 2048,
    dtype: DTypeLike = np.float32,
    norm: Optional[float] = None,
) -> np.ndarray:
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = normalize(window, norm=norm) ** 2
    win_sq = pad_center(win_sq, size=n_fft)

    # Fill the envelope
    __window_ss_fill(x, win_sq, n_frames, hop_length)

    return x


###############################################################################
# Functions
###############################################################################
def stft(
    y: np.ndarray,
    *,
    n_fft: int = 2048,
    window: np.ndarray,
    dtype: Optional[DTypeLike] = None,
) -> np.ndarray:
    # Set the default hop
    hop_length = int(n_fft // 4)

    # Pad the window out to n_fft size
    fft_window = pad_center(window, size=n_fft)

    # Reshape so that the window can be broadcast
    fft_window = expand_to(fft_window, ndim=1 + y.ndim, axes=-2)

    # Pad the time series so that frames are centered
    if n_fft > y.shape[-1]:
        warnings.warn(
            f"n_fft={n_fft} is too large for input signal of length={y.shape[-1]}"
        )

    # Set up the padding array to be empty, and we'll fix the target dimension later
    padding = [(0, 0) for _ in range(y.ndim)]

    # How many frames depend on left padding?
    start_k = int(np.ceil(n_fft // 2 / hop_length))

    # What's the first frame that depends on extra right-padding?
    tail_k = (y.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

    if tail_k <= start_k:
        # If tail and head overlap, then just copy-pad the signal and carry on
        start = 0
        extra = 0
        y_frames_pre = None
        y_frames_post = None

        padding[-1] = (n_fft // 2, n_fft // 2)
        y = np.pad(y, padding, mode=PAD_MODE)
    else:
        # If tail and head do not overlap, then we can implement padding on each part separately
        # and avoid a full copy-pad

        # "Middle" of the signal starts here, and does not depend on head padding
        start = start_k * hop_length - n_fft // 2
        padding[-1] = (n_fft // 2, 0)

        # +1 here is to ensure enough samples to fill the window
        # fixes bug #1567
        y_pre = np.pad(
            y[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
            padding,
            mode=PAD_MODE,
        )
        y_frames_pre = frame(y_pre, frame_length=n_fft, hop_length=hop_length)
        # Trim this down to the exact number of frames we should have
        y_frames_pre = y_frames_pre[..., :start_k]

        # How many extra frames do we have from the head?
        extra = y_frames_pre.shape[-1]

        # Determine if we have any frames that will fit inside the tail pad
        if tail_k * hop_length - n_fft // 2 + n_fft <= y.shape[-1] + n_fft // 2:
            padding[-1] = (0, n_fft // 2)
            y_post = np.pad(
                y[..., (tail_k) * hop_length - n_fft // 2:], padding, mode=PAD_MODE
            )
            y_frames_post = frame(
                y_post, frame_length=n_fft, hop_length=hop_length
            )
            # How many extra frames do we have from the tail?
            extra += y_frames_post.shape[-1]
        else:
            # In this event, the first frame that touches tail padding would run off
            # the end of the padded array
            # We'll circumvent this by allocating an empty frame buffer for the tail
            # this keeps the subsequent logic simple
            post_shape = list(y_frames_pre.shape)
            post_shape[-1] = 0
            y_frames_post = np.empty_like(y_frames_pre, shape=post_shape)

    if dtype is None:
        dtype = dtype_r2c(y.dtype)

    # Window the time series.
    y_frames = frame(y[..., start:], frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    shape = list(y_frames.shape)

    # This is our frequency dimension
    shape[-2] = 1 + n_fft // 2

    # If there's padding, there will be extra head and tail frames
    shape[-1] += extra

    stft_matrix = np.zeros(shape, dtype=dtype, order="F")

    # Fill in the warm-up
    if extra > 0 and y_frames_pre is not None and y_frames_post is not None:
        off_start = y_frames_pre.shape[-1]
        stft_matrix[..., :off_start] = scipy.fft.rfft(fft_window * y_frames_pre, axis=-2)

        off_end = y_frames_post.shape[-1]
        if off_end > 0:
            stft_matrix[..., -off_end:] = scipy.fft.rfft(fft_window * y_frames_post, axis=-2)
    else:
        off_start = 0

    n_columns = int(
        MAX_MEM_BLOCK // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize)
    )
    n_columns = max(n_columns, 1)

    for bl_s in range(0, y_frames.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, y_frames.shape[-1])

        stft_matrix[..., bl_s + off_start:bl_t + off_start] = scipy.fft.rfft(
            fft_window * y_frames[..., bl_s:bl_t], axis=-2
        )
    return stft_matrix


def istft(
    stft_matrix: np.ndarray,
    *,
    n_fft: int = 2048,
    window: np.ndarray,
    length: int,
    dtype: Optional[DTypeLike] = None,
) -> np.ndarray:
    win_length = n_fft
    hop_length = int(win_length // 4)

    # Pad out to match n_fft, and add broadcasting axes
    ifft_window = pad_center(window, size=n_fft)
    ifft_window = expand_to(ifft_window, ndim=stft_matrix.ndim, axes=-2)

    # For efficiency, trim STFT frames according to signal length if available
    if length:
        padded_length = length + 2 * (n_fft // 2)
        n_frames = min(stft_matrix.shape[-1], int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[-1]

    if dtype is None:
        dtype = dtype_c2r(stft_matrix.dtype)

    shape = list(stft_matrix.shape[:-2])
    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    if length:
        expected_signal_len = length

    shape.append(expected_signal_len)

    y = np.zeros(shape, dtype=dtype)

    start_frame = int(np.ceil((n_fft // 2) / hop_length))

    # Do overlap-add on the head block
    ytmp = ifft_window * scipy.fft.irfft(stft_matrix[..., :start_frame], n=n_fft, axis=-2)

    shape[-1] = n_fft + hop_length * (start_frame - 1)
    head_buffer = np.zeros(shape, dtype=dtype)

    __overlap_add(head_buffer, ytmp, hop_length)

    # If y is smaller than the head buffer, take everything
    if y.shape[-1] < shape[-1] - n_fft // 2:
        y[..., :] = head_buffer[..., n_fft // 2:y.shape[-1] + n_fft // 2]
    else:
        # Trim off the first n_fft//2 samples from the head and copy into target buffer
        y[..., :shape[-1] - n_fft // 2] = head_buffer[..., n_fft // 2:]

    # This offset compensates for any differences between frame alignment
    # and padding truncation
    offset = start_frame * hop_length - n_fft // 2

    n_columns = int(
        MAX_MEM_BLOCK // (np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize)
    )
    n_columns = max(n_columns, 1)

    frame = 0
    for bl_s in range(start_frame, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * scipy.fft.irfft(stft_matrix[..., bl_s:bl_t], n=n_fft, axis=-2)

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[..., frame * hop_length + offset:], ytmp, hop_length)

        frame += bl_t - bl_s

    # Normalize by sum of squared window
    ifft_window_sum = window_sumsquare(
        window=window,
        n_frames=n_frames,
        n_fft=n_fft,
        hop_length=hop_length,
        dtype=dtype,
    )

    start = n_fft // 2

    ifft_window_sum = fix_length(ifft_window_sum[..., start:], size=y.shape[-1])

    approx_nonzero_indices = ifft_window_sum > tiny(ifft_window_sum)

    y[..., approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    return y
