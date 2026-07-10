###############################################################################
# Global Imports
###############################################################################
import os
from functools import lru_cache
from pathlib import Path

###############################################################################
# 3PP Imports
###############################################################################
import torch

###############################################################################
# Local Imports
###############################################################################
from teamml_audio_aug.core._advanced import rms, minmax


###############################################################################
# Constants
###############################################################################
SUPPORTED_EXTENSIONS = (
    ".wav",
)


###############################################################################
# Config
###############################################################################
_env_sr = os.getenv("TEAM_ML_SR")
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
    paths: list[Path | str] | Path | str,
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


def calculate_rms(samples: torch.Tensor):
    """Given a numpy array of audio samples, return its Root Mean Square (RMS)."""
    return torch.mean(rms(samples))


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Equal-gain or equal-energy (within ~1%) cross-fade mask pair with
    smooth start and end.
    https://signalsmith-audio.co.uk/writing/2021/cheap-energy-crossfade/
    """
    x = torch.linspace(0, 1, length, dtype=torch.float32)
    x2 = 1 - x
    a = x * x2
    k = 1.4186 if equal_energy else -0.70912
    b = a * (1 + k * a)
    c = b + x
    d = b + x2
    fade_in = c * c
    fade_out = d * d
    return fade_in, fade_out


def get_max_abs_amplitude(samples: torch.Tensor):
    min_amplitude, max_amplitude = minmax(samples)
    max_abs_amplitude = torch.max(torch.abs(min_amplitude), torch.abs(max_amplitude))
    return max_abs_amplitude
