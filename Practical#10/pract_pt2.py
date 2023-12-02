import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


def find_pitch(freq_dom, freq_range):
    peak_index = np.argmax(np.abs(freq_dom))
    pitch = freq_range[peak_index]
    return pitch


def shift_pitch(freq_dom, desired_shift, Fs):
    shift_bins = int(desired_shift * time_dom.shape[0] / Fs)

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


Fs, raw = wavfile.read("Adele.wav")
time_dom = np.array(raw)

block_size = 4096
output_signal = []
target_pitch = 440

for i in range(0, time_dom.shape[0], block_size):
    block = time_dom[i : i + block_size]

    freq_dom = np.fft.fft(block)
    freq_range = np.abs(np.fft.fftfreq(block.shape[0], d=1 / Fs))

    pitch = find_pitch(freq_dom, freq_range)
    print(pitch)
    pitch_shift_amount = target_pitch - pitch

    shifted_freq_dom = shift_pitch(freq_dom, pitch_shift_amount, Fs)

    converted_block = np.fft.ifft(shifted_freq_dom)

    output_signal.extend(converted_block)


output_signal = np.array(output_signal).astype("int32")

plt.plot(time_dom)
plt.plot(output_signal)
plt.show()

sample, segment, Zxx = signal.stft(time_dom, fs=Fs)
X, Y = np.meshgrid(segment, sample)
plt.pcolormesh(X, Y, np.log(np.abs(Zxx.real)))
plt.show()
