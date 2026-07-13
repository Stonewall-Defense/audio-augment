# Audio Augmentations

New takes on audio augmentations for TEAM-ML models and data. If you are just stumbling across this package, you should probably use [Audiomentations](https://github.com/iver56/audiomentations) or [torch-audiomentations](https://github.com/iver56/torch-audiomentations).

## Prerequisites

- Python 3.14 runtime
- Pip for package installation

## Installation

```bash
# Pip is the classic
pip install teamml_audio_augment

# Should also work ith `uv`
uv add teamml_audio_augment
```

## Credit Where Credit is Due

- In the [`AirAbsorption` class](teamml_audio_augment/augmentations/air_absorption.py), the `_fft_freq` function is adapted from [`librosa`](https://librosa.org/doc/latest/generated/librosa.fft_frequencies.html)
- In the [`Infrasound` class](teamml_audio_augment/augmentations/infrasound.py), the `_oscillator_bank` function is adapted from [`torchaudio`](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.prototype.functional.oscillator_bank.html)
- In [_advanced.py](teamml_audio_augment/core/_advanced.py), the functions are adapted from several advanced and/or compiled libraries:
  - The `rms` function is adapted from [`numpy-rms`](https://pypi.org/project/numpy-rms/)
  - The `minmax` function is adapted from [`numpy-minmax`](https://pypi.org/project/numpy-minmax/)
  - The `_next_fast_len` function is adapted from [`scipy.signal`](https://docs.scipy.org/doc/scipy/reference/signal.html)

Our contributions include everything in [advanced](teamml_audio_augment/advanced/) and the `Infrasound` class.

## License

MIT.
