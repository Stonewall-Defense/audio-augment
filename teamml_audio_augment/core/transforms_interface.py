###############################################################################
# Global Imports
###############################################################################
from abc import ABC, abstractmethod
import random
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import numpy as np

###############################################################################
# Local Imports
###############################################################################
from teamml_audio_augment.core.utils import get_default_sample_rate, is_multichannel


###############################################################################
# Exceptions
###############################################################################
class MultichannelAudioNotSupportedException(Exception):
    pass


class MonoAudioNotSupportedException(Exception):
    pass


class WrongMultichannelAudioShape(Exception):
    pass


###############################################################################
# Base Classes
###############################################################################
class BaseWaveformTransform(ABC):
    supports_mono = True
    supports_multichannel = False

    def __init__(self, sample_rate: Optional[int], *, p=0.5):

        self.sample_rate = sample_rate or get_default_sample_rate()
        self.p = p

        assert 0 < self.sample_rate
        assert 0 <= self.p <= 1

        self.should_apply: Optional[bool] = None
        self.are_parameters_frozen = False

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect with the exact same parameters to multiple sounds.
        """
        self.are_parameters_frozen = True

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False

    @abstractmethod
    def apply(self, samples: np.ndarray) -> np.ndarray:
        ...

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        if not self.are_parameters_frozen or self.should_apply is None:
            self.randomize_parameters(samples)
        if self.should_apply and len(samples) > 0:
            if is_multichannel(samples):
                # Note: We multiply by 8 here to allow big batches of very short audio
                if samples.shape[0] > samples.shape[1] * 8:
                    raise WrongMultichannelAudioShape(
                        "Multichannel audio must have channels first, not channels"
                        " last. In other words, the shape must be (channels, samples),"
                        " not (samples, channels). See"
                        " https://iver56.github.io/audiomentations/guides/multichannel_audio_array_shapes/"
                        " for more info."
                    )
                if not self.supports_multichannel:
                    raise MultichannelAudioNotSupportedException(
                        "{} only supports mono audio, not multichannel audio. In other"
                        " words, a 1-dimensional input ndarray was expected, but the"
                        " input had more than 1 dimension.".format(
                            self.__class__.__name__
                        )
                    )
            elif not self.supports_mono:
                raise MonoAudioNotSupportedException(
                    "{} only supports multichannel audio, not mono audio".format(
                        self.__class__.__name__
                    )
                )
            return self.apply(samples)
        return samples

    def randomize_parameters(self, samples: np.ndarray):
        self.should_apply = random.random() < self.p
