###############################################################################
# 3PP Imports
###############################################################################
from audiomentations import Compose
import click
import torch
import torchaudio

###############################################################################
# Certus Imports
###############################################################################
from audiofeatures import load_input, FeatureChannel, FeatureSource
from plot_spec import plot_with_time_domain

###############################################################################
# Local Imports
###############################################################################
from augment.Infrasound import Infrasound, NoiseLevelType


###############################################################################
# Contants
###############################################################################
SAMPLE_RATE = 22_050    # Hz
DURATION_SEC = 1


###############################################################################
# Config
###############################################################################
feature_channels = [
    FeatureChannel(SAMPLE_RATE, n_fft=1024, hop_length=512, is_logarithmic=True, is_mel=True),
]
FEATURE_SOURCE = FeatureSource(feature_channels, stack_spectra=True)


###############################################################################
# ! MAIN
###############################################################################
@click.group()
def cli():
    pass


@cli.command("rel")
@click.option("--duration_secs", default=DURATION_SEC)
@click.option("--save", is_flag=True)
def rel(duration_secs: int, save: bool):
    augs = Compose([
        Infrasound(noise_level_type=NoiseLevelType.RELATIVE, min_snr_db=0, max_snr_db=10, p=1.0),
    ])

    wav = load_input("input/rq-voice-color.wav", target_sr=SAMPLE_RATE, duration_secs=duration_secs)
    clean_spec = FEATURE_SOURCE.forward(wav).squeeze(0)

    plot_with_time_domain(clean_spec, wav, SAMPLE_RATE, "Clean Spectrum")

    tmp_wav = wav.squeeze(0).numpy()
    aug_raw = augs(tmp_wav, SAMPLE_RATE)
    tmp_aug = torch.from_numpy(aug_raw)
    aug_spec = FEATURE_SOURCE.forward(tmp_aug)

    plot_with_time_domain(aug_spec, tmp_aug, SAMPLE_RATE, "Augmented Spectrum")

    if save:
        torchaudio.save("output/rel.wav", tmp_aug.unsqueeze(0), SAMPLE_RATE, bits_per_sample=16, encoding="PCM_S")


@cli.command("abs")
@click.option("--save", is_flag=True)
def abs(save: bool):
    augs = Compose([
        Infrasound(noise_level_type=NoiseLevelType.ABSOLUTE, min_absolute_rms_db=-20, max_absolute_rms_db=-10, p=1.0),
    ])

    wav = torch.zeros(SAMPLE_RATE * DURATION_SEC)
    clean_spec = FEATURE_SOURCE.forward(wav).squeeze(0)

    plot_with_time_domain(clean_spec, wav, SAMPLE_RATE, "Empty Spectrum")

    tmp_wav = wav.squeeze(0).numpy()
    aug_raw = augs(tmp_wav, SAMPLE_RATE)
    tmp_aug = torch.from_numpy(aug_raw)
    aug_spec = FEATURE_SOURCE.forward(tmp_aug)

    plot_with_time_domain(aug_spec, tmp_aug, SAMPLE_RATE, "Infrasound Spectrum")

    if save:
        torchaudio.save("output/abs.wav", tmp_aug.unsqueeze(0), SAMPLE_RATE, bits_per_sample=16, encoding="PCM_S")


if __name__ == "__main__":
    cli()
