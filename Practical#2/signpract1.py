import numpy as np
import matplotlib.pyplot as plt

def Ui(lowerBound, upperBound, shift):
    time_step = np.arange (lowerBound , upperBound , 1 ) # Time s t e p
    impulse_signal = np.zeros_like (time_step) # Im pul se s i g n a l
    for i in range (len (time_step)) :
        if time_step [i] == shift :
            impulse_signal [i] = 1 # Se t im p ul se l o c a t i o n t o 1 in Y a x i s

    return [time_step, impulse_signal]


def Us(lowerBound, upperBound, shift):
    time_step = np.arange (lowerBound , upperBound , 1 ) # Time s t e p
    step_signal = np.zeros_like (time_step) # Im pul se s i g n a l
    for i in range (len (time_step)) :
        if time_step [i] >= shift :
            step_signal [i] = 1 # Se t im p ul se l o c a t i o n t o 1 in Y a x i s

    return [time_step, step_signal]

def Ur(lowerBound, upperBound, shift):
    time_step = np.arange (lowerBound , upperBound , 1 ) # Time s t e p
    ramp_signal = np.zeros_like (time_step) # Im pul se s i g n a l
    for i in range (len (time_step)) :
        if time_step [i] >= shift :
            ramp_signal [i] = time_step[i] - shift # Se t im p ul se l o c a t i o n t o 1 in Y a x i s

    return [time_step, ramp_signal]

[n, i1] = Ui (-20,21,0)
[_, i2] = Ui (-20,21,-1)
[_, i3] = Ui (-20,21,1)

W = 2* i1 - i2 - i3

# 2 a
#2 * Ui[n] - Ui[n+1] - Ui [n-1]

[n, s1] = Us (-20,21,0)
[_, s2] = Us (-20,21,1)
[_, s3] = Us (-20,21,4)
X = s1 - 2* s2 +s3
# 4 a
# Us[n] - 2* Ui[n-1] - Ui [n-4]

# 4 b
[n, i1] = Ui (-20,21,-1)
[_, i2] = Ui (-20,21,0)
[_, s3] = Us (-20,21,-1)
[_, s4] = Us (-20,21,2)
Y = i1-i2+s3-s4

# 4 c
Q = (-1/2)**n * s1

[n, r1] = Ur (-20,21,0)

[_, i61] = Ui(-20,21,5)

#6 a
[n, r61] = Ur (-20,21,-2)
[_, s61] = Ui(-20,21,0)
[_, s62] = Ui(-20,21,5)

Z = r61 - 2*s61 - n * s62

# 6 b
# e^(n-7) - n*Ui[n-5] - 3*Ur[n]
WW = np.exp(n-7) - n*i61 - 3*r1
plt.stem(n, WW)
plt.show()


### task 7 
'''
Multiplying a sequence with a function would result in a scaled or shifted version of the original sequence.
If we multiply the sine function sin(n) with an impulse sequence Ui[n],
the result would be a sequence that is zero everywhere except at n=0,
where it takes the value of sin(0) = 0.
This is because the impulse sequence is zero everywhere except at n=0,
where it takes the value of 1. So, the product of sin(n) and Ui[n] would be sin(n)*Ui[n] = sin(0)*Ui[0] = 0.

'''