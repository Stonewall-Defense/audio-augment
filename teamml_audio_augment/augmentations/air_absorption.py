###############################################################################
# Global Imports
###############################################################################
import logging
import random
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import numpy as np
import scipy.signal

###############################################################################
# Local Imports
###############################################################################
from teamml_audio_augment.core.transforms_interface import BaseWaveformTransform
from teamml_audio_augment.core.fft import stft, istft


###############################################################################
# Config
###############################################################################
LOGGER = logging.getLogger("teamMl")


###############################################################################
# Helpers
###############################################################################
def _next_power_of_2(x: int) -> int:
    """
    taken jhoyla's answer here:
    https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _get_temperature_humidity_key(temperature: float, humidity: float) -> str:
    key = str(int(temperature)) + "C"
    bounds = [30, 50, 70, 90]
    for n in range(1, len(bounds)):
        if bounds[n - 1] <= humidity <= bounds[n]:
            key += f"_{bounds[n-1]}-{bounds[n]}%"
            break
    return key


def _fft_freq(sr: int, n_fft: int = 2048):
    # Adapted from librosa: https://librosa.org/doc/latest/generated/librosa.fft_frequencies.html
    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)


###############################################################################
# Exports
###############################################################################
class AirAbsorption(BaseWaveformTransform):
    """
    Apply a Lowpass-like filterbank with variable octave attenuation that simulates attenuation of
    higher frequencies due to air absorption. This transform is parametrized by temperature,
    humidity, and the distance between audio source and microphone.

    This is not a scientifically accurate transform but basically applies a uniform
    filterbank with attenuations given by:

    att = exp(- distance * absorption_coefficient)

    where distance is the microphone-source assumed distance in meters and `absorption_coefficient`
    is adapted from a lookup table by pyroomacoustics [1]. It can also be seen as a lowpass filter
    with variable octave attenuation.

    Note: This only "simulates" the damping of high frequencies, and does not
    attenuate according to the distance law. Gain augmentation needs to be done separately.

    [1] https://github.com/LCAV/pyroomacoustics
    """

    supports_multichannel = True

    # Table of air absorption coefficients adapted from `pyroomacoustics`.
    # The keys are of the form:
    #   "<degrees>C_<minimum_humidity>-<maximum_humidity>%"
    #
    # And the values are attenuation coefficients `coef` that attenuate the corresponding band
    # in "center_freq" by exp(-coef * <microphone-source distance>).
    # The original table does not have the last two columns which have been extrapolated from the
    # pyroomacoustics table using `curve_fit`
    air_absorption_table = {
        "10C_30-50%": [
            x * 1e-3 for x in [0.1, 0.2, 0.5, 1.1, 2.7, 9.4, 29.0, 91.5, 289.0]
        ],
        "10C_50-70%": [
            x * 1e-3 for x in [0.1, 0.2, 0.5, 0.8, 1.8, 5.9, 21.1, 76.6, 280.2]
        ],
        "10C_70-90%": [
            x * 1e-3 for x in [0.1, 0.2, 0.5, 0.7, 1.4, 4.4, 15.8, 58.0, 214.9]
        ],
        "20C_30-50%": [
            x * 1e-3 for x in [0.1, 0.3, 0.6, 1.0, 1.9, 5.8, 20.3, 72.3, 259.9]
        ],
        "20C_50-70%": [
            x * 1e-3 for x in [0.1, 0.3, 0.6, 1.0, 1.7, 4.1, 13.5, 44.4, 148.7]
        ],
        "20C_70-90%": [
            x * 1e-3 for x in [0.1, 0.3, 0.6, 1.1, 1.7, 3.5, 10.6, 31.2, 93.8]
        ],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000],
    }

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        min_temperature: float = 10.0,
        max_temperature: float = 20.0,
        min_humidity: int = 30,
        max_humidity: int = 90,
        min_distance: float = 10.0,
        max_distance: float = 100.0,
        p=0.5,
    ):
        """
        :param min_temperature: Minimum temperature in Celsius (can take a value of either 10.0 or 20.0)
        :param max_temperature: Maximum temperature in Celsius (can take a value of either 10.0 or 20.0)
        :param min_humidity: Minimum humidity in percent (between 30 and 90)
        :param max_humidity: Maximum humidity in percent (between 30 and 90)
        :param min_distance: Minimum microphone-source distance in meters.
        :param max_distance: Maximum microphone-source distance in meters.
        :param p: The probability of applying this transform
        """
        if float(min_temperature) not in [10.0, 20.0]:
            raise ValueError("min_temperature must be either 10.0 or 20.0")
        if float(max_temperature) not in [10.0, 20.0]:
            raise ValueError("max_temperature must be either 10.0 or 20.0")
        if min_temperature > max_temperature:
            raise ValueError("min_temperature must not be greater than max_temperature")
        if not (30 <= min_humidity <= max_humidity <= 90):
            raise ValueError(
                "min_humidity and max_humidity must be in range [30, 90], and"
                " min_humidity must not be greater than max_humidity"
            )
        if min_distance <= 0.0:
            raise ValueError("min_distance must be > 0.0")
        if min_distance > 2500.0:
            raise ValueError("min_distance must be <= 2500.0")
        if max_distance <= 0.0:
            raise ValueError("max_distance must be > 0.0")
        if max_distance > 2500.0:
            raise ValueError("max_distance must be <= 2500.0")
        if min_distance > max_distance:
            raise ValueError("min_distance must not be greater than max_distance")

        super().__init__(sample_rate, p=p)

        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

        self.min_humidity = min_humidity
        self.max_humidity = max_humidity

        self.min_distance = min_distance
        self.max_distance = max_distance

        self.temperature = self.min_temperature
        self.humidity = self.min_humidity
        self.distance = self.min_distance

    def randomize_parameters(self, samples: np.ndarray):
        super().randomize_parameters(samples)
        self.temperature = 10 * random.randint(
            int(self.min_temperature) // 10, int(self.max_temperature) // 10
        )
        self.humidity = random.randint(
            self.min_humidity, self.max_humidity
        )
        self.distance = random.uniform(self.min_distance, self.max_distance)

    def apply(self, samples: np.ndarray) -> np.ndarray:
        # Choose correct absorption coefficients
        key = _get_temperature_humidity_key(
            self.temperature, self.humidity
        )

        # Convert to attenuations
        attenuation_values = np.exp(
            -self.distance * np.array(self.air_absorption_table[key])
        )

        # Calculate n_fft so that the lowest band can be stored in a single
        # fft bin.
        first_band_bw = self.air_absorption_table["center_freqs"][0] / (2**0.5)
        n_fft = _next_power_of_2(int(self.sample_rate / 2 / first_band_bw))
        window = scipy.signal.get_window("hann", n_fft, fftbins=True)

        # Frequencies to calculate the attenuations caused by air absorption
        frequencies = _fft_freq(sr=self.sample_rate, n_fft=n_fft)

        # Interpolate to the desired frequencies (we have to do this in dB)
        db_target_attenuations = np.interp(
            frequencies,
            self.air_absorption_table["center_freqs"],
            20 * np.log10(attenuation_values),
        )

        linear_target_attenuations = 10 ** (db_target_attenuations / 20)

        # Apply using STFT
        if len(samples.shape) == 1:
            _stft = stft(samples, window=window, n_fft=n_fft)

            # Compute mask
            mask = np.tile(linear_target_attenuations, (_stft.shape[1], 1)).T

            # Compute target degraded audio
            result = istft(_stft * mask, window=window, length=len(samples))
        else:
            result = np.zeros_like(samples, dtype=samples.dtype)

            for chn_idx, channel in enumerate(samples):
                _stft = stft(channel,
                             n_fft=n_fft,
                             window=window,
                             )

                # Compute mask
                mask = np.tile(linear_target_attenuations, (_stft.shape[1], 1)).T

                # Compute target degraded audio
                result[chn_idx, :] = istft(_stft * mask,
                                           n_fft=n_fft,
                                           window=window,
                                           length=result.shape[1],
                                           )

        LOGGER.debug(f"Appied air absorption: {self.temperature} deg / {self.humidity} %RH / {self.distance} m")

        return result
