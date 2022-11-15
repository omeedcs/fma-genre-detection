import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa #  python package for music and audio analysis


#modified from starter code of https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
def get_spectrogram(waveform, n_fft = 400, win_len = None, hop_len = None,power = 2.0):  
    spectrogram = T.Spectrogram(
      n_fft=n_fft,
      win_length=win_len,
      hop_length=hop_len,
      center=True,
      pad_mode="reflect",
      power=power)
    return spectrogram(waveform)


# https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
def get_mel_spectrogram(waveform, sample_rate, n_fft = 400, win_len = None, hop_len = 512, n_mels=128):
    mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_len,
    hop_length=hop_len,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    onesided=True,
    n_mels=n_mels,
    mel_scale="htk")
    return mel_spectrogram(waveform)

# https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
def plot_spectrogram(spectrogram, title = "Spectrogram Plot for Waveform", xlabel = "Frequency Bins", ylabel = "Frame"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    im = axs.imshow(librosa.power_to_db(spectrogram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)