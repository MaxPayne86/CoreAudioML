import numpy as np
import matplotlib.pyplot as plot
import torch
#from CoreAudioML.networks import AsymmetricAdvancedClip as clipper
#from CoreAudioML.networks import AdvancedClip as clipper
#from CoreAudioML.networks import StandardCubicClip as clipper
from CoreAudioML.networks import AsymmetricStandardCubicClip as clipper

# threshold 0.1 - 0.9
def advanced_clip(samples, threshold):
    out = []
    for In in samples:
        theta = (np.abs(In) - threshold) / (1 - threshold)
        if np.abs(In) >= threshold:
            out_ = ((In * threshold + (1 - threshold)) * np.tanh(theta))
            out.append(out_)
        else:
            out.append(In)
    return np.array(out)

# alpha 0.1 - 10
def standard_cubic(samples, alpha):
    out = []
    for In in samples:
        x = In * (1 / alpha)
        if x <= -1:
            fx = -2/3
        elif x >= 1:
            fx = 2/3
        else:
            fx = x - (np.power(x, 3) / 3)
        out.append(fx * alpha)
    return np.array(out)

# alpha1, alpha2 0.1 - 10
def asymmetric_standard_cubic(samples, alpha1, alpha2):
    out = []
    for In in samples:
        if In > 0:
            alpha = alpha1
        else:
            alpha = alpha2
        x = In * (1 / alpha)
        if x <= -1:
            fx = -2/3
        elif x >= 1:
            fx = 2/3
        else:
            fx = x - (np.power(x, 3) / 3)
        out.append(fx * alpha)
    return np.array(out)

start_time = 0
end_time = 0.004 # To show 4 periods
sample_rate = 48000
time = np.arange(start_time, end_time, 1/sample_rate)
theta = 0
frequency = 1000
amplitude = 1
sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)

clip = clipper()
sinewave_t = torch.tensor(sinewave)
with torch.no_grad():
    clip.bias[0] = torch.tensor(0.9)
    clip.bias[1] = torch.tensor(10)
    #clip.bias[0] = torch.tensor(0.5)
    clip_out_t = clip(sinewave_t)
clip_out1 = clip_out_t.cpu().data.numpy()

#clip_out2 = advanced_clip(sinewave, 0.5)

#clip_out2 = standard_cubic(sinewave, 0.5)

clip_out2 = asymmetric_standard_cubic(sinewave, 0.9, 10)

plot.plot(time, sinewave, color='g', label='sin')
plot.title('Clipper output with sine input')
plot.xlabel('Time')
plot.ylabel('Amplitude')
plot.plot(time, clip_out1, color='r', label='torch')
plot.plot(time, clip_out2, color='b', label='numpy')
plot.legend()
plot.grid(True, which='both')
plot.savefig('test_clipper.png')
