###############################################################################
# Global Imports
###############################################################################
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import numpy as np
import numpy_minmax
import numpy_rms
import soundfile
from tinytag import TinyTag

###############################################################################
# Constants
###############################################################################
SUPPORTED_EXTENSIONS = (
    ".wav",
)


###############################################################################
# Config
###############################################################################
_env_sr = os.getenv("TEAMML_AUDIOAUG_SR")
_DEFAULT_SAMPLE_RATE = int(_env_sr) if _env_sr is not None else 44_100


def set_default_sample_rate(sr: int):
    global _DEFAULT_SAMPLE_RATE
    _DEFAULT_SAMPLE_RATE = sr


def get_default_sample_rate():
    return _DEFAULT_SAMPLE_RATE


###############################################################################
# Exports
###############################################################################
def find_audio_files(
    root_path,
    filename_endings=SUPPORTED_EXTENSIONS,
    traverse_subdirectories=True,
    follow_symlinks=True,
):
    """Return a list of paths to all audio files with the given extension(s) in a directory.
    Also traverses subdirectories by default.
    """
    file_paths = []

    for root, _, filenames in os.walk(root_path, followlinks=follow_symlinks):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)

            if filename.lower().endswith(filename_endings):
                file_paths.append(Path(file_path))
        if not traverse_subdirectories:
            # prevent descending into subfolders
            break

    return file_paths


def find_audio_files_in_paths(
    paths: list[Path] | list[str] | Path | str,
    filename_endings=SUPPORTED_EXTENSIONS,
    traverse_subdirectories=True,
    follow_symlinks=True,
):
    """Return a list of paths to all audio files with the given extension(s) contained in the list or in its directories.
    Also traverses subdirectories by default.
    """

    file_paths = []

    if isinstance(paths, (list, tuple, set)):
        path_lst = list(paths)
    else:
        path_lst = [paths]

    for p in path_lst:
        if str(p).lower().endswith(SUPPORTED_EXTENSIONS):
            file_path = Path(os.path.abspath(p))
            file_paths.append(file_path)
        elif os.path.isdir(p):
            file_paths += find_audio_files(
                p,
                filename_endings=filename_endings,
                traverse_subdirectories=traverse_subdirectories,
                follow_symlinks=follow_symlinks,
            )
    return file_paths


def calculate_rms(samples: np.ndarray):
    """Given a numpy array of audio samples, return its Root Mean Square (RMS)."""
    return np.mean(numpy_rms.rms(samples))


def calculate_desired_noise_rms(clean_rms, snr: float):
    """
    Given the Root Mean Square (RMS) of a clean sound and a desired signal-to-noise ratio (SNR),
    calculate the desired RMS of a noise sound to be mixed in.
    Based on https://github.com/Sato-Kunihiko/audio-SNR/blob/8d2c933b6c0afe6f1203251f4877e7a1068a6130/create_mixed_audio_file.py#L20
    :param clean_rms: Root Mean Square (RMS) - a value between 0.0 and 1.0
    :param snr: Signal-to-Noise (SNR) Ratio in dB - typically somewhere between -20 and 60
    :return:
    """
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms


def convert_decibels_to_amplitude_ratio(decibels: float):
    return 10 ** (decibels / 20)


@lru_cache(maxsize=8)
def get_crossfade_mask_pair(
    length: int, equal_energy: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Equal-gain or equal-energy (within ~1%) cross-fade mask pair with
    smooth start and end.
    https://signalsmith-audio.co.uk/writing/2021/cheap-energy-crossfade/
    """
    x = np.linspace(0, 1, length, dtype=np.float32)
    x2 = 1 - x
    a = x * x2
    k = 1.4186 if equal_energy else -0.70912
    b = a * (1 + k * a)
    c = b + x
    d = b + x2
    fade_in = c * c
    fade_out = d * d
    return fade_in, fade_out


def get_max_abs_amplitude(samples: np.ndarray):
    min_amplitude, max_amplitude = numpy_minmax.minmax(samples)
    max_abs_amplitude = max(abs(min_amplitude), abs(max_amplitude))
    return max_abs_amplitude


def is_multichannel(wave: np.ndarray) -> bool:
    return wave.shape[0] > 1


def _metadata(filename: Path | str):
    if not str(filename).endswith("wav"):
        raise ValueError(f"Only WAV files are supported: {filename}")

    info = TinyTag.get(filename)
    n_chan = info.channels
    sr = info.samplerate

    if n_chan is None or sr is None:
        raise ValueError(f"Could not get metadata for file: {filename}")
    else:
        return n_chan, sr


def load_wav(filename: Path | str,
             *,
             start_sec: Optional[float] = None,
             end_sec: Optional[float] = None,
             ):
    n_chan, sr = _metadata(filename)

    if start_sec and start_sec < 0:
        raise ValueError("start_sec must be at least zero")
    elif end_sec and end_sec < 0:
        raise ValueError("end_sec must be at least zero")
    elif start_sec and end_sec and end_sec <= start_sec:
        raise ValueError("end_sec must be strictly higher than start_sec if both are provided")

    start_samples = int(start_sec * sr * n_chan) if start_sec else 0
    end_samples = int(end_sec * sr * n_chan) if end_sec else None
    frames = (end_samples - start_samples) if end_samples is not None else -1

    audio_raw, _ = soundfile.read(filename, start=start_samples, frames=frames, fill_value=0, always_2d=True)
    audio = audio_raw.astype(np.float32).T

    return np.mean(audio, axis=0, keepdims=True) if is_multichannel(audio) else audio


def mel_to_hz(mel: float):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def hz_to_mel(freq: float):
    return 2595.0 * math.log10(1.0 + (freq / 700.0))
