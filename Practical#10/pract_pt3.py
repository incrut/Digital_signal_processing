import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


def find_pitch(freq_dom, freq_range):
    peak_index = np.argmax(np.abs(freq_dom))
    pitch = freq_range[peak_index]
    return pitch


def shift_pitch(freq_dom, desired_shift, Fs):
    shift_bins = int(desired_shift * freq_dom.shape[0] / Fs)

    shifted_freq_dom_pos = np.roll(freq_dom[: freq_dom.shape[0] // 2], shift_bins)
    shifted_freq_dom_neg = np.roll(freq_dom[freq_dom.shape[0] // 2 :], -shift_bins)

    if shift_bins > 0:
        shifted_freq_dom_pos[:shift_bins] = 0
        shifted_freq_dom_neg[-shift_bins:] = 0
    else:
        shifted_freq_dom_pos[shift_bins:] = 0
        shifted_freq_dom_neg[:-shift_bins] = 0

    shifted_freq_dom = np.concatenate((shifted_freq_dom_pos, shifted_freq_dom_neg))
    return shifted_freq_dom


block_size = 4096
desired_pitch = 440
results = []

Fs, raw = wavfile.read("Adele.wav")
time_dom = np.array(raw)

for i in range(0, time_dom.shape[0], block_size):
    block = time_dom[i : i + block_size]

    freq_dom = np.fft.fft(block)
    freq_range = np.abs(np.fft.fftfreq(block.shape[0], d=1 / Fs))

    pitch = find_pitch(freq_dom, freq_range)
    pitch_difference = desired_pitch - pitch

    shifted_freq_dom = shift_pitch(freq_dom, pitch_difference, Fs)

    converted_block = np.fft.ifft(shifted_freq_dom)

    results.extend(converted_block)

results = np.array(results).astype("int32")

wavfile.write("auto_tuned.wav", Fs, results)

sample, segment, Zxx_original = signal.stft(time_dom, fs=Fs)
sample, segment, Zxx_shifted = signal.stft(results, fs=Fs)

plt.subplot(2, 1, 1)
plt.title("Original Signal")
plt.pcolormesh(segment, sample, np.log(np.abs(Zxx_original.real)))
plt.colorbar()

plt.subplot(2, 1, 2)
plt.title("Shifted Signal")
plt.pcolormesh(segment, sample, np.log(np.abs(Zxx_shifted.real)))
plt.colorbar()
plt.tight_layout()
plt.show()
