import torchaudio.transforms as T
import torch
import torchaudio

def resample(waveform, orig_freq, new_freq):
    sampler = torchaudio.transforms.Resample(orig_freq,new_freq)
    return sampler(waveform)

def truncate_sample(waveform, max_length):
    num_samples = waveform.shape[1]
    if num_samples > max_length:
        waveform=waveform[:,:max_length]
    return waveform

def pad_right(waveform, min_samples):
    raise NotImplementedError()

def convert_one_channel(waveform):
    return torch.mean(waveform, dim=0, keepdim= True)