###############################################################################
# Global Imports
###############################################################################
import random
import warnings
from pathlib import Path
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
from tinytag import TinyTag
import torch

###############################################################################
# Certus Imports
###############################################################################
from AudioMlSpecTools import WavReader

###############################################################################
# Local Imports
###############################################################################
from teamml_audio_aug.core.transforms_interface import BaseWaveformTransform
from teamml_audio_aug.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
    find_audio_files_in_paths,
)


###############################################################################
# Exports
###############################################################################
class AddBackgroundNoise(BaseWaveformTransform):
    def __init__(
        self,
        sounds_path: list[Path | str] | Path | str,
        *,
        sample_rate: Optional[int] = None,
        min_snr_db: float = 3.0,
        max_snr_db: float = 30.0,
        p: float = 0.5,
        lru_cache_size: int | None = None,
    ):
        super().__init__(sample_rate, p=p)

        # Boundary parameters
        self.sounds_path = sounds_path
        self.sound_file_paths = find_audio_files_in_paths(self.sounds_path)
        self.sound_file_paths = [str(p) for p in self.sound_file_paths]

        assert len(self.sound_file_paths) > 0

        if min_snr_db > max_snr_db:
            raise ValueError("min_snr_db must not be greater than max_snr_db")
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        # Changing parameters
        self.snr_db = self.min_snr_db
        self.noise_file_path = self.sound_file_paths[0]
        self.offset = 0.0
        self.duration = 0.0

        # Helper parameters
        if lru_cache_size is not None:
            raise ValueError(
                "Passing lru_cache_size is no longer supported, as the cache has been removed (since v0.43.0)."
            )
        self.time_info_arr = torch.full((len(self.sound_file_paths),), -1.0)

    def randomize_parameters(self, samples: torch.Tensor):
        super().randomize_parameters(samples)

        if self.should_apply:
            self.snr_db = random.uniform(self.min_snr_db, self.max_snr_db)

            file_idx = random.randint(0, len(self.sound_file_paths) - 1)
            self.noise_file_path = self.sound_file_paths[file_idx]

            if self.time_info_arr[file_idx] == -1.0:
                duration = TinyTag.get(self.noise_file_path).duration or 0.0
                self.time_info_arr[file_idx] = duration

            noise_duration = float(self.time_info_arr[file_idx])
            signal_duration = len(samples) / self.sample_rate

            min_noise_offset = 0.0
            max_noise_offset = max(0.0, noise_duration - signal_duration)

            self.offset = random.uniform(min_noise_offset, max_noise_offset)
            self.duration = signal_duration

    def apply(self, samples: torch.Tensor) -> torch.Tensor:
        end_sec = self.offset + self.duration
        noise_sound = WavReader(target_sr=self.sample_rate).read(self.noise_file_path, start_sec=self.offset, end_sec=end_sec)

        noise_rms = calculate_rms(noise_sound)
        if noise_rms < 1e-9:
            warnings.warn(
                "The file {} is too silent to be added as noise. Returning the input"
                " unchanged.".format(self.noise_file_path)
            )
            return samples

        clean_rms = calculate_rms(samples)

        desired_noise_rms = calculate_desired_noise_rms(clean_rms, self.snr_db)

        # Adjust the noise to match the desired noise RMS
        noise_sound = noise_sound * (desired_noise_rms / noise_rms)

        # Repeat the sound if it shorter than the input sound
        num_samples = len(samples)
        while len(noise_sound) < num_samples:
            noise_sound = torch.concatenate((noise_sound, noise_sound))

        if len(noise_sound) > num_samples:
            noise_sound = noise_sound[0:num_samples]

        # Return a mix of the input sound and the background noise sound
        return samples + noise_sound
