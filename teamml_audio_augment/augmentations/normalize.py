###############################################################################
# Global Imports
###############################################################################
import logging
from typing import Literal, Optional

###############################################################################
# 3PP Imports
###############################################################################
import torch

###############################################################################
# Local Imports
###############################################################################
from teamml_audio_augment.core.transforms_interface import BaseWaveformTransform
from teamml_audio_augment.core.utils import get_max_abs_amplitude


###############################################################################
# Config
###############################################################################
LOGGER = logging.getLogger("teamMl")


###############################################################################
# Exports
###############################################################################
class Normalize(BaseWaveformTransform):
    """
    Apply a constant amount of gain, so that highest signal level present in the sound becomes
    0 dBFS, i.e. the loudest level allowed if all samples must be between -1 and 1. Also known
    as peak normalization.
    """

    supports_multichannel = True

    def __init__(self,
                 *,
                 sample_rate: Optional[int] = None,
                 apply_to: Literal["all", "only_too_loud_sounds"] = "all",
                 p: float = 0.5
                 ):
        super().__init__(sample_rate, p=p)

        assert apply_to in ("all", "only_too_loud_sounds")
        self.apply_to = apply_to
        self.max_amplitude = 1.0

    def randomize_parameters(self, samples: torch.Tensor):
        super().randomize_parameters(samples)
        if self.should_apply:
            self.max_amplitude = float(get_max_abs_amplitude(samples))

    def apply(self, samples: torch.Tensor) -> torch.Tensor:
        if (self.apply_to == "only_too_loud_sounds" and self.max_amplitude < 1.0):
            return samples

        if self.max_amplitude > 0:
            LOGGER.debug("Applied normalization")
            return samples / self.max_amplitude
        else:
            return samples
