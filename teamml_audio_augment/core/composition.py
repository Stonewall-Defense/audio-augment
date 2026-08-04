###############################################################################
# Global Imports
###############################################################################
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import random
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import numpy as np

###############################################################################
# Certus Imports
###############################################################################
from teamml_audio_augment.core.transforms_interface import BaseWaveformTransform


###############################################################################
# Interfaces
###############################################################################
class BaseCompose(ABC):
    def __init__(self,
                 transforms: Sequence[BaseWaveformTransform | BaseCompose],
                 *,
                 p: float = 1.0,
                 shuffle: bool = False,
                 ):
        self.transforms = transforms
        self.p = p
        self.shuffle = shuffle
        self.are_parameters_frozen = False

    @abstractmethod
    def __call__(self, samples: np.ndarray, apply_to_children: bool) -> np.ndarray:
        ...

    def randomize_parameters(self, samples: np.ndarray, apply_to_children=True):
        """
        Randomize and define parameters of every transform in composition.
        """
        if apply_to_children:
            for transform in self.transforms:
                transform.randomize_parameters(samples)

    def freeze_parameters(self, apply_to_children=True):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect chain with the exact same parameters to multiple
        sounds.
        """
        self.are_parameters_frozen = True
        if apply_to_children:
            for transform in self.transforms:
                transform.freeze_parameters()

    def unfreeze_parameters(self, apply_to_children=True):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False
        if apply_to_children:
            for transform in self.transforms:
                transform.unfreeze_parameters()


###############################################################################
# Classes
###############################################################################
class Compose(BaseCompose):
    """
    Compose applies the given sequence of transforms when called,
    optionally shuffling the sequence for every call.

    Usage example:

    ```
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
    ])

    # Generate 2 seconds of dummy audio for the sake of example
    samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)

    # Augment/transform/perturb the audio data
    augmented_samples = augment(samples=samples, sample_rate=16000)
    ```
    """

    def __init__(self,
                 transforms: Sequence[BaseWaveformTransform | BaseCompose],
                 *,
                 p: float = 1.0,
                 shuffle: bool = False,
                 ):
        super().__init__(transforms, p=p, shuffle=shuffle)

    def __call__(self, samples: np.ndarray, apply_to_children=True) -> np.ndarray:
        transforms = list(self.transforms).copy()
        should_apply = random.random() < self.p

        # TODO: Adhere to self.are_parameters_frozen
        # https://github.com/iver56/audiomentations/issues/135

        if should_apply:
            if self.shuffle:
                random.shuffle(transforms)
            for transform in transforms:
                samples = transform(samples) if isinstance(transform, BaseWaveformTransform) else transform(samples, apply_to_children)

        return samples


class SomeOf(BaseCompose):
    """
    SomeOf randomly picks several of the given transforms when called, and applies these
    transforms. The number of transforms to apply can be chosen in two different ways:

        - Pick exactly n transforms:
            Example:    # pick exactly two of the transforms
                        SomeOf(2, [transform1, transform2, transform3])

        - Pick between a minimum and maximum number of transforms.
            Examples:   # pick 1 to 3 of the transforms
                        SomeOf((1, 3), [transform1, transform2, transform3])

                        # pick 2 to all of the transforms
                        SomeOf((2, None), [transform1, transform2, transform3, transform4])

    Usage example:
    ```
    augment = SomeOf(
        (2, None),
        [
            TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
            PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
            Gain(min_gain_in_db=-12, max_gain_in_db=-6, p=1.0),
        ],
    )

    # Generate 2 seconds of dummy audio for the sake of example
    samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)

    # Augment/transform/perturb the audio data
    augmented_samples = augment(samples=samples, sample_rate=16000)

    # Result: 2 or more transforms will be applied from the list of transforms.
    ```
    """

    def __init__(self,
                 num_transforms: int | tuple[int, Optional[int]],
                 transforms: Sequence[BaseWaveformTransform | BaseCompose],
                 *,
                 p: float = 1.0):
        super().__init__(transforms, p=p)

        self.transform_indexes = []
        self.num_transforms = num_transforms
        self.should_apply = True

    def randomize_parameters(self, samples: np.ndarray, apply_to_children=True):
        super().randomize_parameters(samples, apply_to_children)

        self.should_apply = random.random() < self.p
        if self.should_apply:
            if isinstance(self.num_transforms, tuple):
                if self.num_transforms[1] is None:
                    num_transforms_to_apply = random.randint(
                        self.num_transforms[0], len(self.transforms)
                    )
                else:
                    num_transforms_to_apply = random.randint(
                        self.num_transforms[0], self.num_transforms[1]
                    )
            else:
                num_transforms_to_apply = self.num_transforms

            all_transforms_indexes = list(range(len(self.transforms)))
            self.transform_indexes = sorted(
                random.sample(all_transforms_indexes, num_transforms_to_apply)
            )

    def __call__(self, samples: np.ndarray, apply_to_children=False) -> np.ndarray:
        if not self.are_parameters_frozen:
            apply_to_children = False
            self.randomize_parameters(samples, apply_to_children)

        if self.should_apply:
            for transform_index in self.transform_indexes:
                tr = self.transforms[transform_index]
                samples = tr(samples) if isinstance(tr, BaseWaveformTransform) else tr(samples, apply_to_children)

            return samples

        return samples


class OneOf(BaseCompose):
    """
    OneOf randomly picks one of the given transforms when called, and applies that
    transform. Optional `weights` can be supplied to guide the probability of each transform being chosen.
    Usage example:
    ```
    augment = OneOf([
        TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
        PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
    ], weights=[0.2, 0.8])
    # Generate 2 seconds of dummy audio for the sake of example
    samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)
    # Augment/transform/perturb the audio data
    augmented_samples = augment(samples=samples, sample_rate=16000)
    # Result: The audio was either time-stretched (less likely) or pitch-shifted (4x more likely), but not both
    ```
    """

    def __init__(
        self,
        transforms: Sequence[BaseWaveformTransform | BaseCompose],
        *,
        p: float = 1.0,
    ):
        super().__init__(transforms, p=p)

        self.next_transform = self.transforms[0]
        self.should_apply = True

    def randomize_parameters(self, *args, **kwargs):
        super().randomize_parameters(*args, **kwargs)
        self.should_apply = random.random() < self.p

        if self.should_apply:
            self.next_transform = random.choice(self.transforms)

    def __call__(self, samples: np.ndarray, apply_to_children=False) -> np.ndarray:
        if not self.are_parameters_frozen:
            apply_to_children = False
            self.randomize_parameters(samples, apply_to_children)

        if self.should_apply:
            return self.next_transform(samples) if isinstance(self.next_transform, BaseWaveformTransform) else self.next_transform(samples, apply_to_children)
        else:
            return samples
