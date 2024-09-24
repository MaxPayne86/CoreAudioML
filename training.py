import torch
import torch.nn as nn

from auraloss.perceptual import FIRFilter


class auraloss_adapter(nn.Module):
  '''
  Courtesy of KaisKermani <kaiskermani@gmail.com>

  This is because auraloss loss functions don't take the input in the same shape as torch LSTM Layers do:
  time domain losses input: (seq_len, chs, bs)
  freq domain losses input: (bs, chs, seq_len)

  IDK this is how managed to make it work..
  '''
  def __init__(self, loss_func, domain="time", filter=None):
    super().__init__()
    self.loss_func = loss_func
    self.permutation = None
    if domain == 'time':
      self.permutation = (1, 2, 0)
    elif domain == 'freq':
      self.permutation = (0, 2, 1)

    self.filter = None
    if filter is not None:
      self.filter = FIRFilter(filter)

  def forward(self, input, target):
    alt_input = input.permute(self.permutation)
    alt_target = target.permute(self.permutation)

    if self.filter is not None:
      alt_input, alt_target = self.filter(alt_input, alt_target)

    return self.loss_func(alt_input, alt_target)


# ESR loss calculates the Error-to-signal between the output/target
class ESRLoss(nn.Module):
    def __init__(self):
        super(ESRLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output, target, pooling=False, scale_factor=0.01):
        loss = torch.add(target, -output)
        loss = torch.pow(loss, 2)
        if pooling:
            size = int(list(loss.size())[0])
            pooling_size = int(size * scale_factor)
            # Apply 1D pooling and then interpolate to match the size
            m = nn.AdaptiveAvgPool1d(pooling_size)
            loss = torch.permute(loss, (2, 1, 0))
            out = m(loss)
            loss = nn.functional.interpolate(out, size, mode='linear', align_corners=False)
            loss = torch.permute(loss, (2, 1, 0))
            energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        else:
            loss = torch.mean(loss)
            energy = torch.mean(torch.pow(target, 2)) + self.epsilon

        loss = torch.div(loss, energy)
        return loss


class DCLoss(nn.Module):
    def __init__(self):
        super(DCLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output, target):
        loss = torch.pow(torch.add(torch.mean(target, 0), -torch.mean(output, 0)), 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss


class PreEmph(nn.Module):
    def __init__(self, filter_type='hp', fs=48000):
        super(PreEmph, self).__init__()
        if filter_type == 'hp':
            a1 = (5.9659e+03 * 2 * 3.1416) / fs # Desired hp f = 5.9659e+03 Hz
            self.preemph = FIRFilter(filter_type='hp', coef=a1, fs=fs)
        elif filter_type == 'fd':
            a1 = (5.9659e+03 * 2 * 3.1416) / fs # Desired hp f = 5.9659e+03 Hz
            self.preemph = FIRFilter(filter_type='fd', coef=a1, fs=fs)
        elif filter_type == 'aw':
            # Standard A-weghting
            self.preemph = FIRFilter(filter_type=filter_type, coef=None, fs=fs)
        elif filter_type == 'awlp':
            # [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922)
            # A-weighting with low-pass filter
            self.preemph = FIRFilter(filter_type='aw', coef=None, fs=fs)
            a1 = (5.9659e+03 * 2 * 3.1416) / fs # Desired lp f = 5.9659e+03 Hz
            self._preemph = FIRFilter(filter_type='hp', coef=-a1, fs=fs, ntaps=3) # Note: hp with -coef = lp see Auraloss impl.

    def forward(self, output, target):
        if self._preemph:
            output, target = self.preemph(output.permute(1, 2, 0), target.permute(1, 2, 0))
            output, target = self._preemph(output, target)
        else:
            output, target = self.preemph(output.permute(1, 2, 0), target.permute(1, 2, 0))
        return output.permute(2, 0, 1), target.permute(2, 0, 1)


class LossWrapper(nn.Module):
    def __init__(self, losses, pre_filt=None, samplerate: int = 48000):
        super(LossWrapper, self).__init__()
        loss_dict = {'ESR': ESRLoss(), 'DC': DCLoss()}
        if pre_filt:
            pre_filt = PreEmph(filter_type=pre_filt, fs=samplerate)
            loss_dict['ESRPre'] = lambda output, target: loss_dict['ESR'].forward(*pre_filt(output, target))
        loss_functions = [[loss_dict[key], value] for key, value in losses.items()]

        self.loss_functions = tuple([items[0] for items in loss_functions])
        try:
            self.loss_factors = tuple(torch.Tensor([items[1] for items in loss_functions]))
        except IndexError:
            self.loss_factors = torch.ones(len(self.loss_functions))

    def forward(self, output, target):
        loss = 0
        for i, losses in enumerate(self.loss_functions):
            loss += torch.mul(losses(output, target), self.loss_factors[i])
        return loss


class TrainTrack(dict):
    def __init__(self):
        self.update({'current_epoch': 0, 'training_losses': [], 'validation_losses': [], 'train_av_time': 0.0,
                     'val_av_time': 0.0, 'total_time': 0.0, 'best_val_loss': 1e12, 'test_loss': 0})

    def restore_data(self, training_info):
        self.update(training_info)

    def train_epoch_update(self, loss, ep_st_time, ep_end_time, init_time, current_ep):
        if self['train_av_time']:
            self['train_av_time'] = (self['train_av_time'] + ep_end_time - ep_st_time) / 2
        else:
            self['train_av_time'] = ep_end_time - ep_st_time
        self['training_losses'].append(loss)
        self['current_epoch'] = current_ep
        self['total_time'] += ((init_time + ep_end_time - ep_st_time)/3600)

    def val_epoch_update(self, loss, ep_st_time, ep_end_time):
        if self['val_av_time']:
            self['val_av_time'] = (self['val_av_time'] + ep_end_time - ep_st_time) / 2
        else:
            self['val_av_time'] = ep_end_time - ep_st_time
        self['validation_losses'].append(loss)
        if loss < self['best_val_loss']:
            self['best_val_loss'] = loss
