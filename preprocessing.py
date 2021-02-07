import glob
import os
from pathlib import Path
from typing import Tuple, List
import torch
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB
from torch.nn.functional import pad
from torch import Tensor, mean


def preprocess(audio_path: str, spectrogram_path: str, folds: List[int], resample_rate: int, mel_filters: int, seconds: int):
    os.mkdir(spectrogram_path)
    for fold in folds:
        file_names = audio_files_in_fold(fold, audio_path)
        audio_data = [load_audio(name, fold, audio_path) for name in file_names]
        spectrogram_data = [spectrogram_from_audio(audio, sample_rate, resample_rate, mel_filters, seconds)
                            for audio, sample_rate in audio_data]
        os.mkdir(os.path.normpath(f"{spectrogram_path}/fold{fold}"))
        for spectrogram, name in zip(spectrogram_data, file_names):
            save_spectrogram(spectrogram, name, fold, spectrogram_path)


def audio_files_in_fold(fold: int, audio_path: str) -> List[str]:
    paths = [os.path.normpath(path) for path in glob.glob(f"{audio_path}/fold{fold}/*.wav")]
    return [Path(path).stem for path in paths]


def load_audio(name: str, fold: int, audio_path: str) -> Tuple[Tensor, int]:
    path = os.path.normpath(f"{audio_path}/fold{fold}/{name}.wav")
    return torchaudio.load(path, out=None, normalization=True)


def spectrogram_from_audio(audio: Tensor, sample_rate: int, resample_rate: int, mel_filters: int, seconds: int) -> Tensor:
    resampled_audio = Resample(orig_freq=sample_rate, new_freq=resample_rate)(audio)
    mono_audio = mean(resampled_audio, dim=0, keepdim=True)
    mel_transform = MelSpectrogram(sample_rate=resample_rate, n_mels=mel_filters)
    spectrogram = mel_transform(mono_audio)
    log_spectrogram = AmplitudeToDB()(spectrogram)
    original_length = log_spectrogram.shape[2]
    length = seconds * (resample_rate // mel_transform.hop_length)
    return pad(log_spectrogram, (0, length - original_length)) if original_length < length \
        else log_spectrogram[:, :, :length]


def save_spectrogram(spectrogram: Tensor, name: str, fold: int, spectrogram_path: str):
    path = os.path.normpath(f"{spectrogram_path}/fold{fold}/{name}.pt")
    torch.save(spectrogram, path)

