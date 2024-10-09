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
from torch.autograd.function import custom_fwd, custom_bwd


def wrapperkwargs(func, kwargs):
    return func(**kwargs)


def wrapperargs(func, args):
    return func(*args)


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

    DO NOT USE WIP: https://ez.analog.com/dsp/sigmadsp/f/q-a/570452/asymmetricsoftclipper-and-advancedclip-formulas-are-simply-wrong

    Reference: https://wiki.analog.com/resources/tools-software/sigmastudio/toolbox/nonlinearprocessors/asymmetricsoftclipper

    Implemented by Massimo Pennazio Aida DSP maxipenna@libero.it 2023 All Rights Reserved

    0.1 <= tau1 <= 0.9
    0.1 <= tau2 <= 0.9

    if In > 0:
        if In < tau1:
            Out = In
        else:
            Out = tau1 + (1 - tau1) * tanh( (abs(In) - tau1) / (1 - tau1) )
    else:
        if In < tau2:
            Out = In
        else:
            Out = -tau2 - (1 - tau2) * tanh( (abs(In) - tau2) / (1 - tau2) )

    """
    def __init__(self, size_in=1, size_out=1):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        bias = torch.Tensor(2)
        self.bias = nn.Parameter(bias)
        self.tau_min = 0.1
        self.tau_max = 0.9

        nn.init.uniform_(self.bias, self.tau_min, self.tau_max)  # Bias init

    def forward(self, x):
        tau1 = self.bias.data.clamp(self.tau_min, self.tau_max)[0]
        tau2 = self.bias.data.clamp(self.tau_min, self.tau_max)[1]

        theta2 = torch.div(torch.sub(torch.abs(x), tau2), torch.sub(1, tau2))

        gt_zero = torch.gt(x, 0).type(x.type())
        le_zero = torch.le(x, 0).type(x.type())
        gt_zero_out = torch.mul(gt_zero, x)
        le_zero_out = torch.mul(le_zero, x)

        lt_tau1 = torch.lt(gt_zero_out, tau1).type(x.type())
        ge_tau1 = torch.ge(gt_zero_out, tau1).type(x.type())
        lt_tau1_out = torch.mul(lt_tau1, gt_zero_out)
        ge_tau1_out = torch.mul(ge_tau1, gt_zero_out)
        theta1 = torch.div(torch.sub(torch.abs(ge_tau1_out), tau1), torch.sub(1, tau1))
        f_ge_tau1_out = torch.add(tau1, torch.mul(torch.sub(1, tau2), torch.tanh(theta1)))
        gt_zero_block_out = torch.add(lt_tau1_out, f_ge_tau1_out)

        lt_tau2 = torch.lt(le_zero_out, tau2).type(x.type())
        ge_tau2 = torch.ge(le_zero_out, tau2).type(x.type())
        lt_tau2_out = torch.mul(lt_tau2, le_zero_out)
        ge_tau2_out = torch.mul(ge_tau2, le_zero_out)
        theta2 = torch.div(torch.sub(torch.abs(ge_tau2_out), tau2), torch.sub(1, tau2))
        f_ge_tau2_out = torch.sub(torch.mul(tau2, -1), torch.mul(torch.sub(1, tau2), torch.tanh(theta2)))
        le_zero_block_out = torch.add(lt_tau2_out, f_ge_tau2_out)

        out = torch.add(gt_zero_block_out, le_zero_block_out)
        return out


class AdvancedClip(nn.Module):
    """
    A simple advanced clip unit (tanh)

    DO NOT USE WIP: https://ez.analog.com/dsp/sigmadsp/f/q-a/570452/asymmetricsoftclipper-and-advancedclip-formulas-are-simply-wrong

    Reference: https://wiki.analog.com/resources/tools-software/sigmastudio/toolbox/nonlinearprocessors/advancedclip

    Implemented by Massimo Pennazio Aida DSP maxipenna@libero.it 2023 All Rights Reserved

    0.1 <= threshold <= 0.9

    theta = (abs(In) - threshold) / (1 - threshold)
    if In < threshold:
       Out = In
     else
       Out = (In * threshold + (1 - threshold) * tanh(theta))

    """
    def __init__(self, size_in=1, size_out=1):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        bias = torch.Tensor(1)
        self.bias = nn.Parameter(bias)
        self.thr_min = 0.1
        self.thr_max = 0.9

        nn.init.uniform_(self.bias, self.thr_min, self.thr_max)  # Bias init

    def forward(self, x):
        thr = self.bias.data.clamp(self.thr_min, self.thr_max)
        theta = torch.div(torch.sub(torch.abs(x), thr), torch.sub(1, thr))
        sub_thr = torch.lt(x, thr).type(x.type())
        sub_thr_out = torch.mul(sub_thr, x)
        over_thr = torch.ge(x, thr).type(x.type())
        f_out = torch.add(torch.mul(x, thr), torch.mul(torch.sub(1, thr), torch.tanh(theta)))
        over_thr_out = torch.mul(over_thr, f_out)
        out = torch.add(sub_thr_out, over_thr_out)
        return out
