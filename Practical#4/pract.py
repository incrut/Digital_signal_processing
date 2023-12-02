import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import firwin 

(Fs, raw) = wavfile.read ("Adele.wav")
time_dom  = np.array(raw)

nqyuist_freq = Fs / 2.0
freq = 500 #Hz
filter_order = 1001
f = freq / nqyuist_freq
filter_kernel = firwin(filter_order, f, pass_zero=False)

filter_order = 1001
x = np.arange(-filter_order/2, filter_order/2)  
T = 100
filter_kernel = np.sin(2*np.pi/T*x)/x

filter_kernel /= np.sum(np.abs(filter_kernel))

con_time = np.convolve(time_dom, filter_kernel, mode = "same")

plt.subplot(311)
plt.plot(filter_kernel)

plt.subplot(312)
plt.plot(time_dom[2000:3000])


plt.subplot(313)
plt.plot(con_time[2000:3000])
plt.show()



wavfile.write("out.wav", Fs, con_time.astype('int32'))
