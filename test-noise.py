import torchaudio

from audiofeatures import load_input

from augment.augmentations.AddGaussianNoise import AddGaussianNoise
from augment.augmentations.MadBackgroundNoise import MadClass, create_mad_augment
from augment.augmentations.EscBackgroundNoise import EscMetaClass, create_esc_augment


SAMPLE_RATE = 22050


gauss = AddGaussianNoise(min_snr_db=3, max_snr_db=3, p=1.0)
mad_guns = create_mad_augment(
    "/home/ryan/Desktop/TEAM-ML/datasets/military_audio_dataset/files",
    MadClass.GUNSHOT,
    sample_rate=SAMPLE_RATE,
    min_snr_db=3,
    max_snr_db=3,
    p=1.0,
)
mad_other = create_mad_augment(
    "/home/ryan/Desktop/TEAM-ML/datasets/military_audio_dataset/files",
    [MadClass.COMMUNICATION, MadClass.FIGHTER, MadClass.VEHICLE],
    sample_rate=SAMPLE_RATE,
    min_snr_db=3,
    max_snr_db=3,
    p=1.0,
)
esc_env = create_esc_augment(
    "/home/ryan/Desktop/TEAM-ML/datasets/ESC-50-master/audio",
    [EscMetaClass.NATURAL_SOUNDSCAPES],
    sample_rate=SAMPLE_RATE,
    min_snr_db=3,
    max_snr_db=3,
    p=1.0,
)

samples = load_input("./rq-voice-color.wav", target_sr=SAMPLE_RATE)
with_gauss = gauss(samples)
with_guns = mad_guns(samples)
with_mad_other = mad_other(samples)
with_esc = esc_env(samples)
with_all = esc_env(mad_other(mad_guns(gauss(samples))))

torchaudio.save("with_gauss.wav", with_gauss, SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)
torchaudio.save("with_guns.wav", with_guns, SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)
torchaudio.save("with_mad_other.wav", with_mad_other, SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)
torchaudio.save("with_esc.wav", with_esc, SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)
torchaudio.save("with_all.wav", with_all, SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)
