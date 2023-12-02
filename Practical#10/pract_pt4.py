import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


def find_closest_pitch(freq, note_freqs):
    return note_freqs[np.argmin(np.abs(note_freqs - freq))]


def shift_pitch_stft(freq_dom, desired_shift):
    shifted_freq_dom = np.zeros_like(freq_dom)

    for i in range(freq_dom.shape[0]):
        shifted_freq_dom[i] = np.roll(freq_dom[i], int(desired_shift))

    return shifted_freq_dom


def auto_tune_stft(input_file, output_file, desired_notes):
    Fs, raw = wavfile.read(input_file)
    time_dom = np.array(raw)

    block_size = 4096
    results = []

    sample, segment, Zxx = signal.stft(time_dom, fs=Fs)

    freq_range = np.fft.fftfreq(block_size, d=1 / Fs)
    note_freqs = np.array([440.0 * (2 ** ((n - 69) / 12)) for n in desired_notes])

    for i in range(segment.shape[0]):
        freq_dom = Zxx[:, i]
        pitch = np.abs(freq_range[np.argmax(np.abs(freq_dom))])

        closest_pitch = find_closest_pitch(pitch, note_freqs)
        pitch_difference = closest_pitch - pitch

        shifted_freq_dom = shift_pitch_stft(freq_dom, pitch_difference)

        converted_block = np.fft.irfft(shifted_freq_dom)

        results.extend(converted_block)

    results = np.array(results).astype("int16")
    wavfile.write(output_file, Fs, results)

    plt.subplot(2, 1, 1)
    plt.title("Original Signal")
    plt.pcolormesh(segment, sample, np.log(np.abs(Zxx.real)))
    plt.colorbar()

    shifted_sample, shifted_segment, Zxx_shifted = signal.stft(results, fs=Fs)

    plt.subplot(2, 1, 2)
    plt.title("Shifted Signal")
    plt.pcolormesh(shifted_segment, shifted_sample, np.log(np.abs(Zxx_shifted.real)))
    plt.colorbar()

    plt.show()


input_file = ...
output_file = ...
desired_notes = [69, 71, 73, 76]  # MIDI note numbers for A4, B4, C#5, E5

auto_tune_stft(input_file, output_file, desired_notes)
