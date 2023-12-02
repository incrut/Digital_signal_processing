import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd


new_sample_rate_4 = 1000
sample_rate = 44100
time = 5
frames = sample_rate * time

time_dom = sd.rec(frames, sample_rate, channels=1, dtype = "int32")
sd.wait()

new_sample_rate = sample_rate/2
sd.play(time_dom, new_sample_rate)
sd.wait()