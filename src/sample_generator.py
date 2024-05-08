# implement a python API rather than a command line tool

from collections import OrderedDict
from copy import deepcopy
import os
import random
import pytorch_lightning as pl
import logging
import os
from scipy import rand
from PIL import Image
from torch._C import device

os.environ["HYDRA_FULL_ERROR"] = "1"
from argparse import Namespace
import torch
import hydra
from omegaconf import DictConfig
import argparse
import glob
import librosa

try:
    from networks.vqvae2 import VQVAE
except ImportError:
    from src.networks.vqvae2 import VQVAE

import tqdm
import numpy as np

"""
API demo:

```{python}
m = SampleGenerator(model_path) # load model using a local checkpoint file
sample = m.load_sample(path_to_wav) # preprocess a torch.tensor sample from .wav, .npy, .png

# encoding and decoding
quant_top, quant_bottom, diff, id_top, id_bottom = model.encode(sample)
reconstructed_sample = model.decode(quant_top,quant_bottom)'

# encode and decode in one step:
reconstructed_sample = encode_decode(sample)

# convert generated sample (np.array formatted spectrogram) back to audio
# using Griffin-Lim algorithm to estimate phase
signal, sr = model.to_audio_signal(reconstructed_sample)

# feature-space noise addition
noise_sample = m.add_feature_space_noise(sample,strength=40)

# feature-space interpolation
s1 = m.load_sample(path_to_wav1)
s2 = m.load_sample(path_to_wav2)
interpolated_sample = m.feature_space_interpolation(s1,s2,ratio=.5)
```
"""


class SampleGenerator:
    def __init__(self, model_path, device=None):
        """Initialize the SampleGenerator with a VQVAE model

        Args:
            model_path: str, path to model weights
                if None, initializes an untrained model!
            device: str, device to use. Defaults to None.
                if None, uses cuda:0 or mps if available, else cpu
        """

        # use gpu if available
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        model = VQVAE(in_channel=1)
        if model_path is not None:
            model = load_model(model, model_path, device=device)
        self.model = model

    def encode(self, sample):
        """Encode a sample into the feature space

        Args:
            - sample: torch.tensor eg self.load_sample(path)

        Returns: (guessing at the meaning of these)
            - quant_top: torch.tensor, quantized feature map
            - quant_bottom: torch.tensor, quantized feature map
            - id_top: torch.tensor, index of closest codebook vector
            - id_bottom: torch.tensor, index of closest codebook vector
        """
        sample = sample.to(self.device)
        quant_top, quant_bottom, diff, id_top, id_bottom = self.model.encode(sample)
        return (quant_top, quant_bottom, id_top, id_bottom)

    def load_audio(self, audio_path, sr=16384, seconds=4):
        audio, _sr = librosa.load(audio_path)
        audio = librosa.resample(y=audio, orig_sr=_sr, target_sr=sr)
        audio = librosa.util.fix_length(data=audio, size=seconds)
        return audio

    def load_sample_spectrogram(
        self, audio_path, window_length=16384 * 4, sr=16384, n_fft=1024
    ):
        audio = self.load_audio(audio_path, sr, window_length)
        # features = librosa.feature.melspectrogram(y=audio, n_fft=n_fft)
        features = spec(y=audio, n_fft=n_fft, power=2)
        features = librosa.power_to_db(features)

        if features.shape[0] % 2 != 0:
            features = features[1:, :]
        if features.shape[1] % 2 != 0:
            features = features[:, 1:]
        return features

    def load_sample(self, filepath):
        print(filepath)
        if filepath.endswith(".npy"):
            spectrogram = np.load(filepath)
            spectrogram = np.pad(spectrogram, [(0, 0), (0, 4)], mode="edge")
        elif filepath.endswith(".png"):
            spectrogram = np.array(Image.open(filepath))
        elif filepath.endswith(".wav"):
            spectrogram = self.load_sample_spectrogram(filepath)
        else:
            raise NotImplementedError(f"Filetype {filepath} not supported.")

        # create two leading dimensions
        spectrogram = np.expand_dims(spectrogram, 0)
        spectrogram = np.expand_dims(spectrogram, 0)

        return torch.tensor(spectrogram)

    def decode(self, quant_top, quant_bottom):
        # feature space to image space
        return self.model.decode(quant_top, quant_bottom).detach()

    def to_audio_signal(self, spectrogram, sr=16384):
        """estimate audio signal from spectrogram.

        Uses Griffin-Lim algorithm to estimate phase

        original audio is always resampled to 16384, so that's what we get back also

        Args:
            - spectrogram: np.array
            - sr: int, sample rate of returned signal [Default: 16384]

        Returns:
            - np.array: audio signal at specified sample rate
        """
        spectrogram = librosa.db_to_power(spectrogram)
        signal = librosa.griffinlim(spectrogram)
        if sr != 16384:
            signal = librosa.resample(y=signal, orig_sr=16384, target_sr=sr)

        # normalize
        max = np.max(np.abs(signal))
        if max > 0:
            signal /= max

        return signal, sr

    def add_feature_space_noise(self, sample, strength=0.5):
        """Generate a new sample by adding noise to the feature space

        Args:
            sample: torch.tensor eg self.load_sample(path)
            ratio (float, optional): strength of noise. Defaults to 0.5.

        Returns:
            np.array: reconstructed sample
        """
        q_t, q_b, i_t, i_b = self.encode(sample)
        new_q_t = strength * torch.randn_like(q_t) + q_t
        new_q_b = strength * torch.randn_like(q_b) + q_b
        reconstructed = self.decode(new_q_t, new_q_b).cpu().numpy()[0][0]
        return reconstructed[:, :-4]

    def encode_decode(self, sample):
        """Generate a new sample by adding noise to the feature space

        Args:
            sample: torch.tensor eg self.load_sample(path)

        Returns:
            np.array: reconstructed sample after encoding and decoding
        """
        qant_top, quant_bottom, id_top, id_bottom = self.encode(sample)
        return self.decode(qant_top, quant_bottom).cpu().numpy()[0][0]

    def feature_space_interpolation(self, s1, s2, ratio=0.5):
        """Interpolate between two samples in the feature space

        Args:
            s1: torch.tensor eg self.load_sample(path)
            s2: torch.tensor eg self.load_sample(path)
            ratio (float, optional): interpolation ratio along line segment
                between the two feature vectors. Defaults to 0.5.
                - 0 means s1, 1 means s2
                - values between 0-1 are linear interpolations
                - values outside 0-1 are extrapolations

        Returns:
            np.array: reconstructed sample
        """

        if ratio is None:
            ratio = random.random()  # Generate number in [0,1)

        q_t, q_b, i_t, i_b = self.encode(s1)
        q_t1, q_b1, i_t1, i_b1 = self.encode(s2)
        new_q_t = (q_t1 - q_t) * ratio + q_t
        new_q_b = (q_b1 - q_b) * ratio + q_b

        reconstructed = self.decode(new_q_t, new_q_b).cpu().numpy()[0][0]
        return reconstructed[:, :-4]


def update_model_keys(old_model: OrderedDict, key_to_replace: str = "module."):
    new_model = OrderedDict()
    for key, value in old_model.items():
        if key.startswith(key):
            new_model[key.replace(key_to_replace, "", 1)] = value
        else:
            new_model[key] = value
    return new_model


def load_model(model, model_path, device="cuda:0"):
    weights = torch.load(model_path, map_location="cpu")
    if "model" in weights:
        weights = weights["model"]
    if "state_dict" in weights:
        weights = weights["state_dict"]
    weights = update_model_keys(weights, key_to_replace="net.")
    model.load_state_dict(weights)
    model = model.eval()
    model = model.to(device)
    return model


def spec(y, n_fft, power):
    S = librosa.core.stft(y, n_fft=n_fft)
    return np.abs(S) ** power
