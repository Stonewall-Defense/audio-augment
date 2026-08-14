###############################################################################
# Global Imports
###############################################################################
from copy import copy
import functools
import itertools
import logging
import random
from pathlib import Path
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import numpy as np
from scipy.signal import convolve

###############################################################################
# Local Imports
###############################################################################
from teamml_audio_augment.core.transforms_interface import BaseWaveformTransform
from teamml_audio_augment.core.utils import find_audio_files_in_paths, load_wav


###############################################################################
# Config
###############################################################################
LOGGER = logging.getLogger("teamMl")


###############################################################################
# Exports
###############################################################################
class ApplyImpulseResponse(BaseWaveformTransform):
    """Convolve the audio with a randomly selected impulse response.
    Impulse responses can be created using e.g. http://tulrich.com/recording/ir_capture/
    Impulse responses are represented as audio (ideally wav) files in the given ir_path.
    """

    supports_multichannel = True

    def __init__(
        self,
        ir_path: list[Path] | list[str] | Path | str,
        *,
        sample_rate: Optional[int] = None,
        p=0.5,

        lru_cache_size=128,
        leave_length_unchanged: bool = True,
    ):
        """
        :param ir_path: A path or list of paths to audio file(s) and/or folder(s) with
            audio files. Can be str or Path instance(s). The audio files given here are
            supposed to be impulse responses.
        :param p: The probability of applying this transform
        :param lru_cache_size: Maximum size of the LRU cache for storing impulse response files
        in memory.
        :param leave_length_unchanged: When set to True, the tail of the sound (e.g. reverb at
            the end) will be chopped off so that the length of the output is equal to the
            length of the input.
        """
        super().__init__(sample_rate, p=p)

        self.ir_path = ir_path
        self.ir_files = [str(p) for p in find_audio_files_in_paths(self.ir_path)]
        assert self.ir_files, "No impulse response files found at the specified path."

        self.lru_cache_size = lru_cache_size
        self.load_fn = functools.lru_cache(maxsize=self.lru_cache_size)(self.__load_ir)
        self.leave_length_unchanged = leave_length_unchanged

    def __getstate__(self):
        result = copy(self.__dict__)
        result["load_fn"] = None
        return result

    def __setstate__(self, state):
        self.__dict__ = state
        self.load_fn = functools.lru_cache(maxsize=self.lru_cache_size)(self.__load_ir)

    @staticmethod
    def __load_ir(file_path: Path | str, sample_rate: int):
        return load_wav(file_path)

    def randomize_parameters(self, samples: np.ndarray):
        super().randomize_parameters(samples)
        if self.should_apply:
            self.ir_file_path = random.choice(self.ir_files)

    def apply(self, samples: np.ndarray) -> np.ndarray:
        ir = self.load_fn(self.ir_file_path, self.sample_rate)

        # Expand dimensions to match
        samples_original_dim = samples.ndim
        samples, ir = np.atleast_2d(samples), np.atleast_2d(ir)

        # Preallocate the output array
        output_shape = (samples.shape[0], samples.shape[1] + ir.shape[1] - 1)
        signal_ir = np.empty(output_shape, dtype=samples.dtype)

        # Loop over all samples channels for channelwise convolution
        for i, (sample, impulse_response) in enumerate(zip(samples, itertools.cycle(ir))):
            signal_ir[i, :] = convolve(sample, impulse_response)

        max_value = max(np.amax(signal_ir), -np.amin(signal_ir))
        if max_value > 0.0:
            scale = 0.5 / max_value
            signal_ir *= scale

        if self.leave_length_unchanged:
            signal_ir = signal_ir[..., : samples.shape[-1]]

        # reshape if mono input
        if samples_original_dim == 1:
            signal_ir = signal_ir[0]

        LOGGER.debug(f"Appied impulse response from {self.ir_file_path.split('/')[-1]}")

        return signal_ir
