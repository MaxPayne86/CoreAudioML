import numpy as np
import matplotlib.pyplot as plot
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from CoreAudioML.ddsp import AdvancedClip as advanced_clipper
from CoreAudioML.ddsp import AsymmetricAdvancedClip as asymmetric_advanced_clipper
from CoreAudioML.ddsp import StandardCubicClip as standard_cubic_clipper
from CoreAudioML.ddsp import AsymmetricStandardCubicClip as asymmetric_standard_cubic_clipper

# threshold 0.1 - 0.9
def advanced_clip(samples, threshold):
    out = []
    for In in samples:
        if np.abs(In) <= threshold:
            out.append(In)
        else:
            theta = (np.abs(In) - threshold) / threshold
            if In > 0:
                out_ = threshold * (1 + np.tanh(theta))
            else:
                out_ = -threshold * (1 + np.tanh(theta))
            out.append(out_)
    return np.array(out)

# tau1, tau2 0.1 - 0.9
def asymmetric_advanced_clip(samples, tau1, tau2):
    out = []
    for In in samples:
        if In > 0:
            if In <= tau1:
                out.append(In)
            else:
                theta1 = (np.abs(In) - tau1) / tau1
                out_ = tau1 * (1 + np.tanh(theta1))
                out.append(out_)
        else:
            if np.abs(In) <= tau2:
                out.append(In)
            else:
                theta2 = (np.abs(In) - tau2) / tau2
                out_ = -tau2 * (1 + np.tanh(theta2))
                out.append(out_)
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

sinewave_t = torch.tensor(sinewave)

# Evaluate advanced_clip
with torch.no_grad():
    clip = advanced_clipper()
    clip.weight[0] = torch.tensor(0.5)
    clip.bias[0] = torch.tensor(0)
    clip_out_t = clip(sinewave_t)
clip_out_advanced_torch = clip_out_t.cpu().data.numpy()
clip_out_advanced_numpy = advanced_clip(sinewave, 0.5)

plot.figure()
plot.plot(time, sinewave, color='g', label='sin')
plot.title('Advanced Clip Output: NumPy vs PyTorch')
plot.xlabel('Time')
plot.ylabel('Amplitude')
plot.plot(time, clip_out_advanced_numpy, color='b', label='numpy')
plot.plot(time, clip_out_advanced_torch, color='r', label='torch')
plot.legend()
plot.grid(True, which='both')
plot.savefig('compare_advanced_clipper.png')

# Evaluate standard_cubic
with torch.no_grad():
    clip = standard_cubic_clipper()
    clip.weight[0] = torch.tensor(0.5)
    clip.bias[0] = torch.tensor(0)
    clip_out_t = clip(sinewave_t)
clip_out_standard_torch = clip_out_t.cpu().data.numpy()
clip_out_standard_numpy = standard_cubic(sinewave, 0.5)

plot.figure()
plot.plot(time, sinewave, color='g', label='sin')
plot.title('Standard Cubic Clip Output: NumPy vs PyTorch')
plot.xlabel('Time')
plot.ylabel('Amplitude')
plot.plot(time, clip_out_standard_numpy, color='b', label='numpy')
plot.plot(time, clip_out_standard_torch, color='r', label='torch')
plot.legend()
plot.grid(True, which='both')
plot.savefig('compare_standard_clipper.png')

# Evaluate asymmetric_standard_cubic
with torch.no_grad():
    clip = asymmetric_standard_cubic_clipper()
    clip.weight[0] = torch.tensor(0.9)
    clip.weight[1] = torch.tensor(10)
    clip.bias[0] = torch.tensor(0)
    clip_out_t = clip(sinewave_t)
clip_out_asymmetric_torch = clip_out_t.cpu().data.numpy()
clip_out_asymmetric_numpy = asymmetric_standard_cubic(sinewave, 0.9, 10)

plot.figure()
plot.plot(time, sinewave, color='g', label='sin')
plot.title('Asymmetric Standard Cubic Clip Output: NumPy vs PyTorch')
plot.xlabel('Time')
plot.ylabel('Amplitude')
plot.plot(time, clip_out_asymmetric_numpy, color='b', label='numpy')
plot.plot(time, clip_out_asymmetric_torch, color='r', label='torch')
plot.legend()
plot.grid(True, which='both')
plot.savefig('compare_asymmetric_clipper.png')

# Evaluate asymmetric_advanced_clip
with torch.no_grad():
    clip = asymmetric_advanced_clipper()
    clip.weight[0] = torch.tensor(0.5)
    clip.weight[1] = torch.tensor(0.9)
    clip.bias[0] = torch.tensor(0)
    clip_out_t = clip(sinewave_t)
clip_out_asymmetric_advanced_torch = clip_out_t.cpu().data.numpy()
clip_out_asymmetric_advanced_numpy = asymmetric_advanced_clip(sinewave, 0.5, 0.9)

plot.figure()
plot.plot(time, sinewave, color='g', label='sin')
plot.title('Asymmetric Advanced Clip Output: NumPy vs PyTorch')
plot.xlabel('Time')
plot.ylabel('Amplitude')
plot.plot(time, clip_out_asymmetric_advanced_numpy, color='b', label='numpy')
plot.plot(time, clip_out_asymmetric_advanced_torch, color='r', label='torch')
plot.legend()
plot.grid(True, which='both')
plot.savefig('compare_asymmetric_advanced_clipper.png')
