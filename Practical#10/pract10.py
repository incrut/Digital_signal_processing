import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import chirp


interval_length = 5
sample_rate = 44100
f0 = 440
f1 = 440
t = np.linspace(0, interval_length, int(sample_rate*interval_length))

sweep_signal = chirp(t, f0=f0, f1=f1, t1 = interval_length, method="linear")
sweep_signal *= 1e9

wavfile.write("chirp.wav", sample_rate, sweep_signal.astype("int32"))

plt.plot(sweep_signal)
plt.show()