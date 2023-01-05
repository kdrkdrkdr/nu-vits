from lightning_model import NuWave2
from omegaconf import OmegaConf as OC
import os
import argparse
import datetime
from glob import glob
import torch
import librosa as rosa
from scipy.io.wavfile import write as swrite
import matplotlib.pyplot as plt
from utils.stft import STFTMag
import numpy as np
from scipy.signal import sosfiltfilt
from scipy.signal import butter, cheby1, cheby2, ellip, bessel
from scipy.signal import resample_poly
import random



def main(checkpoint_path:str, wav_path:str, wav_sample_rate:int, steps=8, gt=None, device='cuda'):
    hparams = OC.load('./nuwave2/hparameter.yaml')
    os.makedirs(hparams.log.test_result_dir, exist_ok=True)
    if steps is None or steps == 8:
        steps = 8
        noise_schedule = eval(hparams.dpm.infer_schedule)
    else:
        noise_schedule = None
    model = NuWave2(hparams).to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'] if not('EMA' in checkpoint_path) else ckpt)

    highcut = wav_sample_rate // 2
    nyq = 0.5 * hparams.audio.sampling_rate
    hi = highcut / nyq

    if gt:
        wav, _ = rosa.load(wav_path, sr=hparams.audio.sampling_rate, mono=True)
        wav /= np.max(np.abs(wav))
        wav = wav[:len(wav) - len(wav) % hparams.audio.hop_length]

        order = 8
        sos = cheby1(order, 0.05, hi, btype='lowpass', output='sos')
        wav_l = sosfiltfilt(sos, wav)

        # downsample to the low sampling rate
        wav_l = resample_poly(wav_l, highcut * 2, hparams.audio.sampling_rate)
        # upsample to the original sampling rate
        wav_l = resample_poly(wav_l, hparams.audio.sampling_rate, highcut * 2)

        if len(wav_l) < len(wav):
            wav_l = np.pad(wav, (0, len(wav) - len(wav_l)), 'constant', constant_values=0)
        elif len(wav_l) > len(wav):
            wav_l = wav_l[:len(wav)]
    else:
        wav, _ = rosa.load(wav_path, sr=wav_sample_rate, mono=True)
        wav /= np.max(np.abs(wav))

        # upsample to the original sampling rate
        wav_l = resample_poly(wav, hparams.audio.sampling_rate, wav_sample_rate)
        wav_l = wav_l[:len(wav_l) - len(wav_l) % hparams.audio.hop_length]

    fft_size = hparams.audio.filter_length // 2 + 1
    band = torch.zeros(fft_size, dtype=torch.int64)
    band[:int(hi * fft_size)] = 1

    wav = torch.from_numpy(wav).unsqueeze(0).to(device)
    wav_l = torch.from_numpy(wav_l.copy()).float().unsqueeze(0).to(device)
    band = band.unsqueeze(0).to(device)

    wav_recon, _ = model.inference(wav_l, band, steps, noise_schedule)

    wav_recon = torch.clamp(wav_recon, min=-1, max=1 - torch.finfo(torch.float16).eps)
    swrite(os.path.join(hparams.log.test_result_dir, os.path.split(wav_path)[1]),
           hparams.audio.sampling_rate, wav_recon[0].detach().cpu().numpy())