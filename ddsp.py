# This file is part of CoreAudioML, a project for the course EE-490 at EPFL.
#
# Created by Massimo Pennazio, Aida DSP, 2024 All Rights Reserved
#
# This file contains Google's Magenta DDSP inspired custom PyTorch modules
# that implement specific audio processors.
#
# Checkout https://github.com/magenta/ddsp for more information.

import torch
from torch import nn, Tensor
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


class DifferentiableClamp(Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.

    Ref: https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.

    Ref: https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    """
    return DifferentiableClamp.apply(input, min, max)


def std_cubic(x: Tensor, alpha: Tensor) -> Tensor:
    x = torch.mul(x, torch.div(1, alpha))
    le_one = torch.le(x, -1.0).type(x.type())
    ge_one = torch.ge(x, 1.0).type(x.type())

    gt_one = torch.gt(x, -1.0).type(x.type())
    lt_one = torch.lt(x, 1.0).type(x.type())
    between = torch.mul(gt_one, lt_one)

    le_one_out = torch.mul(le_one, -2/3)
    ge_one_out = torch.mul(ge_one, 2/3)
    between_out = torch.mul(between, x)
    fx = torch.sub(between_out, torch.div(torch.pow(between_out, 3), 3))
    out_ = torch.add(le_one_out, ge_one_out)
    out = torch.mul(torch.add(out_, fx), alpha)
    return out


class AsymmetricStandardCubicClip(nn.Module):
    """
    A simple asymmetric clip unit (standard cubic)

    Reference: https://wiki.analog.com/resources/tools-software/sigmastudio/toolbox/nonlinearprocessors/asymmetricsoftclipper

    Implemented by Massimo Pennazio Aida DSP maxipenna@libero.it 2023 All Rights Reserved

    0.1 <= alpha1 <= 10
    0.1 <= alpha2 <= 10

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
    Out = fx * alpha

    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    bias: Tensor

    def __init__(self, in_features=1, out_features=1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(2, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(1, **factory_kwargs))
        self.alpha_min = 0.1
        self.alpha_max = 10

        nn.init.constant_(self.weight, self.alpha_max) # Weights init
        nn.init.zeros_(self.bias)  # Bias init

    def forward(self, x: Tensor) -> Tensor:
        alpha = dclamp(self.weight, self.alpha_min, self.alpha_max)
        gt_zero = torch.gt(x, 0).type(x.type())
        le_zero = torch.le(x, 0).type(x.type())
        gt_zero_out = std_cubic(x=torch.mul(x, gt_zero), alpha=alpha[0])
        le_zero_out = std_cubic(x=torch.mul(x, le_zero), alpha=alpha[1])
        return torch.add(torch.add(gt_zero_out, le_zero_out), self.bias)


class StandardCubicClip(nn.Module):
    """
    A simple symmetric clip unit (standard cubic)

    Reference: https://wiki.analog.com/resources/tools-software/sigmastudio/toolbox/nonlinearprocessors/standardcubic

    Implemented by Massimo Pennazio Aida DSP maxipenna@libero.it 2023 All Rights Reserved

    0.1 <= alpha <= 10

    x = In * (1 / alpha)
    if x <= -1:
        fx = -2/3
    elif x >= 1:
        fx = 2/3
    else:
        fx = x - (np.power(x, 3) / 3)
    Out = fx * alpha

    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    bias: Tensor

    def __init__(self, in_features=1, out_features=1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(1, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(1, **factory_kwargs))
        self.alpha_min = 0.1
        self.alpha_max = 10

        nn.init.constant_(self.weight, self.alpha_max) # Weights init
        nn.init.zeros_(self.bias)  # Bias init

    def forward(self, x: Tensor) -> Tensor:
        alpha = dclamp(self.weight, self.alpha_min, self.alpha_max)
        return torch.add(std_cubic(x=x, alpha=alpha), self.bias)


class AsymmetricAdvancedClip(nn.Module):
    """
    A simple asymmetric advanced clip unit (tanh)

    Reference:
    - https://wiki.analog.com/resources/tools-software/sigmastudio/toolbox/nonlinearprocessors/asymmetricsoftclipper
    - https://ez.analog.com/dsp/sigmadsp/f/q-a/570452/asymmetricsoftclipper-and-advancedclip-formulas-are-simply-wrong

    Implemented by Massimo Pennazio Aida DSP maxipenna@libero.it 2023 All Rights Reserved

    0.1 <= tau1 <= 0.9
    0.1 <= tau2 <= 0.9

    Pre-calculate:
        theta1 = (abs(In) - tau1) / tau1
        theta2 = (abs(In) - tau2) / tau2

    If In > 0:
        If In <= tau1:
            Out = In
        Else:
            Out = tau1 * (1 + tanh(theta1))
    Else:
        If abs(In) <= tau2:
            Out = In
        Else:
            Out = -tau2 * (1 + tanh(theta2))

    """
    def __init__(self, size_in=1, size_out=1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weight = nn.Parameter(torch.empty(2, **factory_kwargs))  # Two thresholds: tau1 and tau2
        self.bias = nn.Parameter(torch.empty(1, **factory_kwargs))  # Single bias
        self.tau_min = 0.1
        self.tau_max = 0.9

        nn.init.uniform_(self.weight, self.tau_min, self.tau_max)  # Initialize tau1 and tau2
        nn.init.zeros_(self.bias)  # Initialize bias to zero

    def forward(self, x) -> Tensor:
        # Clamp the thresholds to ensure they stay within valid bounds
        tau = dclamp(self.weight, self.tau_min, self.tau_max)
        tau1, tau2 = tau[0], tau[1]

        # Pre-calculate theta for all inputs
        abs_x = torch.abs(x)
        theta1 = torch.div(torch.sub(abs_x, tau1), tau1)
        theta2 = torch.div(torch.sub(abs_x, tau2), tau2)

        # Positive inputs
        gt_zero = torch.gt(x, 0).type(x.type())
        within_tau1 = torch.le(x, tau1).type(x.type())
        over_tau1 = torch.gt(x, tau1).type(x.type())

        within_tau1_out = torch.mul(within_tau1, x)
        over_tau1_out = torch.mul(over_tau1, torch.mul(tau1, torch.add(1, torch.tanh(theta1))))
        positive_out = torch.mul(gt_zero, torch.add(within_tau1_out, over_tau1_out))

        # Negative inputs
        le_zero = torch.le(x, 0).type(x.type())
        within_tau2 = torch.le(abs_x, tau2).type(x.type())
        over_tau2 = torch.gt(abs_x, tau2).type(x.type())

        within_tau2_out = torch.mul(within_tau2, x)
        over_tau2_out = torch.mul(over_tau2, torch.mul(-tau2, torch.add(1, torch.tanh(theta2))))
        negative_out = torch.mul(le_zero, torch.add(within_tau2_out, over_tau2_out))

        # Combine positive and negative outputs and add bias
        out = torch.add(torch.add(positive_out, negative_out), self.bias)
        return out


class AdvancedClip(nn.Module):
    """
    A simple advanced clip unit (tanh)

    Reference:
    - https://wiki.analog.com/resources/tools-software/sigmastudio/toolbox/nonlinearprocessors/advancedclip
    - https://ez.analog.com/dsp/sigmadsp/f/q-a/570452/asymmetricsoftclipper-and-advancedclip-formulas-are-simply-wrong

    Implemented by Massimo Pennazio Aida DSP maxipenna@libero.it 2023 All Rights Reserved

    0.1 <= threshold <= 0.9

    Pre-calculate:
        theta = (abs(In) - threshold) / threshold

    if abs(In) <= threshold:
        Out = In
    else:
        if In > 0:
            Out = threshold * (1 + tanh(theta))
        else:
            Out = -threshold * (1 + tanh(theta))

    """
    def __init__(self, size_in=1, size_out=1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weight = nn.Parameter(torch.empty(1, **factory_kwargs))  # Single threshold
        self.bias = nn.Parameter(torch.empty(1, **factory_kwargs))  # Single bias
        self.thr_min = 0.1
        self.thr_max = 0.9

        nn.init.uniform_(self.weight, self.thr_min, self.thr_max)  # Initialize threshold
        nn.init.zeros_(self.bias)  # Initialize bias to zero

    def forward(self, x) -> Tensor:
        # Clamp the threshold to ensure it stays within valid bounds
        thr = dclamp(self.weight, self.thr_min, self.thr_max)

        # Compute theta
        theta = torch.div(torch.sub(torch.abs(x), thr), thr)

        # Compute outputs for abs(input) <= threshold
        within_thr = torch.le(torch.abs(x), thr).type(x.type())
        within_thr_out = torch.mul(within_thr, x)

        # Compute outputs for abs(input) > threshold
        over_thr = torch.gt(torch.abs(x), thr).type(x.type())
        positive = torch.gt(x, 0).type(x.type())
        negative = torch.le(x, 0).type(x.type())

        positive_out = torch.mul(positive, torch.mul(thr, torch.add(1, torch.tanh(theta))))
        negative_out = torch.mul(negative, torch.mul(-thr, torch.add(1, torch.tanh(theta))))

        over_thr_out = torch.mul(over_thr, torch.add(positive_out, negative_out))

        # Combine the outputs and add bias
        out = torch.add(torch.add(within_thr_out, over_thr_out), self.bias)
        return out
