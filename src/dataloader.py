import os
import csv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
import json
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as TF
import torch
import soundfile as sf
import librosa
from scipy.io import wavfile
import sys


def convert_to_vggss(filename):
    return filename.replace(".mp4", "")


def load_image(path):
    return Image.open(path).convert('RGB')


def load_flow(path):
    return Image.open(path).convert('L')


def load_audio(path, mode):
    duration = 3.0

    samplerate, samples = wavfile.read(path)

    if mode == 'test':
        samp_lower = float(len(samples)//22050)/2 - duration/2
        samp_higher = float(len(samples)//22050)/2 + duration/2

        audio = samples[int(samp_lower*22050):int(samp_higher*22050)]

        audio = audio / 32767

    else:
        audio = samples / 32767

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * duration:
        n = int(samplerate * duration / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * duration)]

    frequencies, times, spectrogram = signal.spectrogram(
        audio, samplerate, nperseg=512, noverlap=274)
    spectrogram = np.log(spectrogram + 1e-7)
    return spectrogram


class BaseDataset(Dataset):
    def __init__(self, mode, img_files, audio_files, img_path, flow_path, audio_path, audio_transform=None):
        super().__init__()
        self.mode = mode
        self.audio_path = audio_path
        self.flow_path = flow_path
        self.img_path = img_path
        self.flow_path = flow_path

        self.audio_files = audio_files
        self.img_files = img_files

        self.audio_transform = audio_transform

        self.img_normalize = transforms.Compose([transforms.ToTensor()])
        self.flow_normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0], [1])])

        self.img_resize = transforms.Compose(
            [transforms.Resize((224, 224), Image.BICUBIC)])

    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_path, self.img_files[idx])
        frame = load_image(imgpath)

        flow_xpath = os.path.join(
            self.flow_path + "flow_x/", self.img_files[idx])
        flow_x = load_flow(flow_xpath)

        flow_ypath = os.path.join(
            self.flow_path + "flow_y/", self.img_files[idx])
        flow_y = load_flow(flow_ypath)

        audiopath = os.path.join(self.audio_path, self.audio_files[idx])
        spectrogram = self.audio_transform(load_audio(audiopath, self.mode))

        if self.mode == 'train':
            frame = self.img_resize(frame)
            flow_x = self.img_resize(flow_x)
            flow_y = self.img_resize(flow_y)

            # Random Crop
            i, j, h, w = transforms.RandomCrop.get_params(
                frame, output_size=(224, 224))
            frame = TF.crop(frame, i, j, h, w)
            flow_x = TF.crop(flow_x, i, j, h, w)
            flow_y = TF.crop(flow_y, i, j, h, w)

            frame = self.img_normalize(frame)
            flow_x = self.flow_normalize(flow_x)
            flow_y = self.flow_normalize(flow_y)

            flow = torch.cat((flow_x, flow_y), dim=0)

        else:
            frame = self.img_resize(frame)
            flow_x = self.img_resize(flow_x)
            flow_y = self.img_resize(flow_y)

            frame = self.img_normalize(frame)
            flow_x = self.flow_normalize(flow_x)
            flow_y = self.flow_normalize(flow_y)

            flow = torch.cat((flow_x, flow_y), dim=0)

            flip = False

        return frame, flow, spectrogram, self.img_files[idx].split('.')[0]

    def __len__(self):
        return len(self.img_files)
