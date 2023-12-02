import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def Ui(lowerBound, upperBound, shift):
    time_step = np.arange (lowerBound , upperBound , 1 ) # Time s t e p
    impulse_signal = np.zeros_like (time_step) # Im pul se s i g n a l
    for i in range (len (time_step)) :
        if time_step [i] == shift :
            impulse_signal [i] = 1 # Se t im p ul se l o c a t i o n t o 1 in Y a x i s

    return [time_step, impulse_signal]

def convolution (A, B):
    A_len = A.shape[0]
    B_len = B.shape[0]

    Y = np.zeros((A_len + B_len - 1))

    for i in range(A_len):
        for j in range (B_len):
            Y[i+j] += A[i] * B[j]

    return Y

def calc_time(A_t, B_t):
    t_min = A_t[0] + B_t[0]
    t_max = A_t[-1] + B_t[-1]
    time = np.arange(t_min, t_max + 1)

    return time

def calc_conv_time(A, B, Fs):
    A_len = A.shape[0]
    B_len = B.shape[0]

    time = np.arange(0, (A_len + B_len - 1) / Fs, 1 / Fs)

    return time

# f_t = np.array([0,1,2,3,4,5,6])
# f = np.array([0,0,1,1,1,0,0])
# g_t = np.array([-4,-3,-2,-1,0,1,2])
# g = np.array([0,0,3,2,1,0,0])
# f_con_g = convolution(f,g)
# f_con_g_t = calc_time(f_t,g_t)

# plt.subplot(311)
# plt.stem(f_t,f)
# plt.subplot(312)
# plt.stem(g_t,g)
# plt.subplot(313)
# plt.stem(f_con_g_t, f_con_g)
# plt.show()

# x_t = np.arange(4)
# x = np.array([1,2,3,-1])
# n, imp = Ui(-20, 20, -2)

# conv = np.convolve(x, imp)
# time = calc_time(x_t, n)

# plt.subplot(311)
# plt.stem(x_t, x)
# plt.subplot(312)
# plt.stem(n, imp)
# plt.subplot(313)
# plt.stem(time,conv)
# plt.show()

#read and save audio file
(Fs, raw) = wavfile.read ("Adele.wav")
time_dom = np.array(raw)

#Convolve a time domain signal with a [0.5, 0.5] filter
kernel = np.array([0.5, 0.5])
convolved_time_dom = np.convolve(time_dom, kernel)

#The audio sample given in moodle is 32 bit*
#wavfile.write("Adele_out_conv.wav", Fs, convolved_time_dom.astype('uint32'))
#for 16 and 32 bit use int16 and int32

# plt.subplot(211)
# plt.plot(time_dom)
# plt.subplot(212)
# plt.plot(convolved_time_dom)
# plt.show()


# 1) x[n] = [1, 2, 3, −1] and g[n] = Ui[n + 2]
x1 = np.array([1, 2, 3, -1])
g1_t, g1 = Ui(-20, 20, -2)
conv1 = convolution(x1, g1)
time1 = calc_time(np.arange(x1.size), g1_t)
plt.stem(time1, conv1)
plt.title('Convolution of x[n] with g[n] = Ui[n + 2]')


# 2) x[n] = [1, 2, 3, −1] and g[n] = Ui[n − 2]
x2 = np.array([1, 2, 3, -1])
g2_t, g2 = Ui(-20, 20, 2)
conv2 = convolution(x2, g2)
time2 = calc_time(np.arange(x2.size), g2_t)
plt.figure()
plt.stem(time2, conv2)
plt.title('Convolution of x[n] with g[n] = Ui[n - 2]')


# 3) x[n] = 2sin[(π/20)*n] and g[n] = [ (1/6),(1/6),(1/6),(1/6),(1/6),(1/6)]
x3 = 2 * np.sin((np.pi / 20) * np.arange(-20, 20))
g3 = np.ones(6) / 6
conv3 = convolution(x3, g3)
time3 = calc_time(np.arange(x3.size), np.arange(g3.size))
plt.figure()
plt.stem(time3, conv3)
plt.title('Convolution of x[n] with g[n] = [ (1/6),(1/6),(1/6),(1/6),(1/6),(1/6)]')

# 4) x[n] = 2sin[(π/2)*n] and g[n] = [ (1/6),(1/6),(1/6),(1/6),(1/6),(1/6)]
x4 = 2 * np.sin((np.pi / 2) * np.arange(-20, 20))
g4 = np.ones(6) / 6
conv4 = convolution(x4, g4)
time4 = calc_time(np.arange(x4.size), np.arange(g4.size))
plt.figure()
plt.stem(time4, conv4)
plt.title('Convolution of x[n] with g[n] = [ (1/6),(1/6),(1/6),(1/6),(1/6),(1/6)]')

# 5) x[n] = 2sin[(π/20)*n] and g[n] = [−1, 1, −1]
x5 = 2 * np.sin((np.pi / 20) * np.arange(-20, 20))
g5 = np.array([-1, 1, -1])
conv5 = convolution(x5, g5)
time5 = calc_time(np.arange(x5.size), np.arange(g5.size)-1)
plt.figure()
plt.stem(time5, conv5)
plt.title('Convolution of x[n] with g[n] = [-1, 1, -1]')

#plt.show()

"""In case 1, the impulse response g[n] shifts the signal x[n] to the right by 2 units.
 In case 2, the impulse response g[n] shifts the signal x[n] to the left by 2 units.
 n case 3, the signal x[n] is filtered with a moving average filter represented by g[n],
 resulting in a smoothed version of the original signal. In case 4, the signal x[n] oscillates more rapidly than in case 3,
 and is again smoothed by the moving average filter.
 In case 5, the signal x[n] is multiplied by a sequence that alternates between -1 and 1, resulting in an alternating pattern."""


filter_lengths = [10, 100, 1000]
filters = [np.ones((length,))/length for length in filter_lengths]

filtered_audios = []
for filt in filters:
    filtered_audio = np.convolve(time_dom, filt)
    filtered_audios.append(filtered_audio)

for i, filtered_audio in enumerate(filtered_audios):
    plt.figure()
    plt.plot(filtered_audio)
    plt.title("Filtered_audio_out")
    plt.show()
    wavfile.write("Filtered_audio_out.wav", Fs, filtered_audio.astype('int16'))

"""The simple averaging filters smooth out the audio signal by averaging the values in a window.
The larger the window size, the smoother the output audio will be.
This can be useful for removing noise or reducing high-frequency components but
may also result in loss of some important details in the audio."""