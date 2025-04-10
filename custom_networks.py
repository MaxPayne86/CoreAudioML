# This file is part of CoreAudioML, a project for the course EE-490 at EPFL.
#
# Created by Massimo Pennazio, Aida DSP, 2024 All Rights Reserved
#
# This file contains custom network models that mix standard RNNs with custom audio processors or input data manipulation.

import torch, torch.nn as nn
from torch import Tensor
import CoreAudioML.miscfuncs as miscfuncs
import math
from contextlib import nullcontext
from ddsp import AsymmetricStandardCubicClip, StandardCubicClip, AsymmetricAdvancedClip, 


def wrapperkwargs(func, kwargs):
    return func(**kwargs)

def wrapperargs(func, args):
    return func(*args)


class AsymmetricAdvancedClipSimpleRNN(nn.Module):
    """
    A simple AdvancedClip RNN class that consists of an asymmetric anvanced clip unit and a single recurrent unit of type LSTM, GRU or Elman, followed by a fully connected
    layer. You can configure the position of the clip unit using clip_position arg.

    Table for clip_position values:

    clip_position = 0x00: Clip unit is at the head of the network
    clip_position = 0x01: Clip unit is at the tail of the network
    clip_position = 0x11: Clip unit is parallel to the recurrent unit
    """
    def __init__(self, input_size=1, output_size=1, unit_type="GRU", hidden_size=12, clip_position=0x00, bias_fl=True,
                 num_layers=1, parallel_clip=False):
        super(AsymmetricAdvancedClipSimpleRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.clip = AsymmetricStandardCubicClip(1, 1)
        self.clip_position = clip_position
        self.rec = wrapperargs(getattr(nn, unit_type), [input_size, hidden_size, num_layers])
        self.lin = nn.Linear(hidden_size, output_size, bias=bias_fl)
        self.bias_fl = bias_fl
        self.save_state = True
        self.hidden = None

    def forward(self, x):
        if self.clip_position == 0x00: # Head pos
            x, self.hidden = self.rec(self.clip(x), self.hidden)
            return self.lin(x)
        elif self.clip_position == 0x01: # Tail pos
            x, self.hidden = self.rec(x, self.hidden)
            return self.clip(self.lin(x))
        elif self.clip_position == 0x11: # Parallel pos
            res = x
            x, self.hidden = self.rec(x, self.hidden)
            return torch.add(self.lin(x), self.clip(res))
        else:
            print("Error! Invalid value for arg clip_position!")
            exit(1)

    # detach hidden state, this resets gradient tracking on the hidden state
    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()

    # changes the hidden state to None, causing pytorch to create an all-zero hidden state when the rec unit is called
    def reset_hidden(self):
        self.hidden = None

    # This functions saves the model and all its paraemters to a json file, so it can be loaded by a JUCE plugin
    def save_model(self, file_name, direc=''):
        if direc:
            miscfuncs.dir_check(direc)
        model_data = {'model_data': {'model': 'AsymmetricAdvancedClipSimpleRNN', 'input_size': self.rec.input_size, 'clip_position': self.clip_position,
                                     'output_size': self.lin.out_features, 'unit_type': self.rec._get_name(),
                                     'num_layers': self.rec.num_layers, 'hidden_size': self.rec.hidden_size,
                                     'bias_fl': self.bias_fl}}

        if self.save_state:
            model_state = self.state_dict()
            for each in model_state:
                model_state[each] = model_state[each].cpu().data.numpy().tolist()
            model_data['state_dict'] = model_state

        miscfuncs.json_save(model_data, file_name, direc)

    # train_epoch runs one epoch of training
    def train_epoch(self, input_data, target_data, loss_fcn, optim, bs, init_len=200, up_fr=1000):
        # shuffle the segments at the start of the epoch
        shuffle = torch.randperm(input_data.shape[1])

        # Iterate over the batches
        ep_loss = 0
        for batch_i in range(math.ceil(shuffle.shape[0] / bs)):
            # Load batch of shuffled segments
            input_batch = input_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]
            target_batch = target_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]

            # Initialise network hidden state by processing some samples then zero the gradient buffers
            self(input_batch[0:init_len, :, :])
            self.zero_grad()

            # Choose the starting index for processing the rest of the batch sequence, in chunks of args.up_fr
            start_i = init_len
            batch_loss = 0
            # Iterate over the remaining samples in the mini batch
            for k in range(math.ceil((input_batch.shape[0] - init_len) / up_fr)):
                # Process input batch with neural network
                output = self(input_batch[start_i:start_i + up_fr, :, :])

                # Calculate loss and update network parameters
                loss = loss_fcn(output, target_batch[start_i:start_i + up_fr, :, :])
                loss.backward()
                optim.step()

                # Set the network hidden state, to detach it from the computation graph
                self.detach_hidden()
                self.zero_grad()

                # Update the start index for the next iteration and add the loss to the batch_loss total
                start_i += up_fr
                batch_loss += loss

            # Add the average batch loss to the epoch loss and reset the hidden states to zeros
            ep_loss += batch_loss / (k + 1)
            self.reset_hidden()
        return ep_loss / (batch_i + 1)

    # Only proc processes a the input data and calculates the loss, optionally grad can be tracked or not
    def process_data(self, input_data, target_data, loss_fcn, chunk, grad=False):
        with (torch.no_grad() if not grad else nullcontext()):
            output = torch.empty_like(target_data)
            for l in range(int(output.size()[0] / chunk)):
                output[l * chunk:(l + 1) * chunk] = self(input_data[l * chunk:(l + 1) * chunk])
                self.detach_hidden()
            # If the data set doesn't divide evenly into the chunk length, process the remainder
            if not (output.size()[0] / chunk).is_integer():
                output[(l + 1) * chunk:-1] = self(input_data[(l + 1) * chunk:-1])
            self.reset_hidden()
            loss = loss_fcn(output, target_data)
        return output, loss


class AdvancedClipSimpleRNN(nn.Module):
    """
    A simple AdvancedClip RNN class that consists of an anvanced clip unit in front of a single recurrent unit of type LSTM, GRU or Elman, followed by a fully connected
    layer
    """
    def __init__(self, input_size=1, output_size=1, unit_type="GRU", hidden_size=12, skip=0, bias_fl=True,
                 num_layers=1):
        super(AdvancedClipSimpleRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.clip = StandardCubicClip(1, 1)
        self.rec = wrapperargs(getattr(nn, unit_type), [input_size, hidden_size, num_layers])
        self.lin = nn.Linear(hidden_size, output_size, bias=bias_fl)
        self.bias_fl = bias_fl
        self.skip = skip
        self.save_state = True
        self.hidden = None

    def forward(self, x, hidden=None):
        x = self.clip(x)
        if self.skip > 0:
            # save the residual for the skip connection
            res = x[:, :, 0:self.skip]
            if(hidden):
                x, self.hidden = self.rec(x, hidden)
                return ((self.lin(x) + res), self.hidden)
            else:
                x, self.hidden = self.rec(x, self.hidden)
                return self.lin(x) + res
        else:
            if(hidden):
                x, self.hidden = self.rec(x, hidden)
                return (self.lin(x), self.hidden)
            else:
                x, self.hidden = self.rec(x, self.hidden)
                return self.lin(x)

    # detach hidden state, this resets gradient tracking on the hidden state
    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()

    # changes the hidden state to None, causing pytorch to create an all-zero hidden state when the rec unit is called
    def reset_hidden(self):
        self.hidden = None

    # This functions saves the model and all its paraemters to a json file, so it can be loaded by a JUCE plugin
    def save_model(self, file_name, direc=''):
        if direc:
            miscfuncs.dir_check(direc)
        model_data = {'model_data': {'model': 'AdvancedClipSimpleRNN', 'input_size': self.rec.input_size, 'skip': self.skip,
                                     'output_size': self.lin.out_features, 'unit_type': self.rec._get_name(),
                                     'num_layers': self.rec.num_layers, 'hidden_size': self.rec.hidden_size,
                                     'bias_fl': self.bias_fl}}

        if self.save_state:
            model_state = self.state_dict()
            for each in model_state:
                model_state[each] = model_state[each].cpu().data.numpy().tolist()
            model_data['state_dict'] = model_state

        miscfuncs.json_save(model_data, file_name, direc)

    # train_epoch runs one epoch of training
    def train_epoch(self, input_data, target_data, loss_fcn, optim, bs, init_len=200, up_fr=1000):
        # shuffle the segments at the start of the epoch
        shuffle = torch.randperm(input_data.shape[1])

        # Iterate over the batches
        ep_loss = 0
        for batch_i in range(math.ceil(shuffle.shape[0] / bs)):
            # Load batch of shuffled segments
            input_batch = input_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]
            target_batch = target_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]

            # Initialise network hidden state by processing some samples then zero the gradient buffers
            self(input_batch[0:init_len, :, :])
            self.zero_grad()

            # Choose the starting index for processing the rest of the batch sequence, in chunks of args.up_fr
            start_i = init_len
            batch_loss = 0
            # Iterate over the remaining samples in the mini batch
            for k in range(math.ceil((input_batch.shape[0] - init_len) / up_fr)):
                # Process input batch with neural network
                output = self(input_batch[start_i:start_i + up_fr, :, :])

                # Calculate loss and update network parameters
                loss = loss_fcn(output, target_batch[start_i:start_i + up_fr, :, :])
                loss.backward()
                optim.step()

                # Set the network hidden state, to detach it from the computation graph
                self.detach_hidden()
                self.zero_grad()

                # Update the start index for the next iteration and add the loss to the batch_loss total
                start_i += up_fr
                batch_loss += loss

            # Add the average batch loss to the epoch loss and reset the hidden states to zeros
            ep_loss += batch_loss / (k + 1)
            self.reset_hidden()
        return ep_loss / (batch_i + 1)

    # Only proc processes a the input data and calculates the loss, optionally grad can be tracked or not
    def process_data(self, input_data, target_data, loss_fcn, chunk, grad=False):
        with (torch.no_grad() if not grad else nullcontext()):
            output = torch.empty_like(target_data)
            for l in range(int(output.size()[0] / chunk)):
                output[l * chunk:(l + 1) * chunk] = self(input_data[l * chunk:(l + 1) * chunk])
                self.detach_hidden()
            # If the data set doesn't divide evenly into the chunk length, process the remainder
            if not (output.size()[0] / chunk).is_integer():
                output[(l + 1) * chunk:-1] = self(input_data[(l + 1) * chunk:-1])
            self.reset_hidden()
            loss = loss_fcn(output, target_data)
        return output, loss


class EnhancedReceptionSimpleRNN(nn.Module):
    """
    A simple EnhancedReception RNN class that consists of a single recurrent unit of type LSTM, GRU or Elman, followed by a fully connected
    layer
    """
    def __init__(self, input_size=1, output_size=1, unit_type="GRU", hidden_size=12, bias_fl=True,
                 num_layers=1, reception_size=8, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(EnhancedReceptionSimpleRNN, self).__init__()
        self.factory_kwargs = factory_kwargs
        self.input_size = input_size
        self.output_size = output_size
        self.reception_size = reception_size
        self.buf = None  # Placeholder for buffer
        self.rec = wrapperargs(getattr(nn, unit_type), [input_size * reception_size, hidden_size, num_layers])
        self.lin = nn.Linear(hidden_size, output_size, bias=bias_fl)
        self.bias_fl = bias_fl
        self.save_state = True
        self.hidden = None

    def forward(self, x: Tensor) -> Tensor:
        if self.buf is None:
            # Initialize buffer based on the input shape
            batch_size, seq_len, num_features = x.shape # num_features = input_size
            self.buf = torch.zeros([batch_size, seq_len, self.reception_size * num_features], **self.factory_kwargs)

        # Append new input to the buffer and remove the oldest elements
        self.buf = torch.cat((self.buf[:, :, num_features:], x), dim=2)
        rec_out, self.hidden = self.rec(self.buf, self.hidden)
        return self.lin(rec_out)

    # detach hidden state, this resets gradient tracking on the hidden state
    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()

    # changes the hidden state to None, causing pytorch to create an all-zero hidden state when the rec unit is called
    def reset_hidden(self):
        self.hidden = None

    # This functions saves the model and all its paraemters to a json file, so it can be loaded by a JUCE plugin
    def save_model(self, file_name, direc=''):
        if direc:
            miscfuncs.dir_check(direc)
        model_data = {'model_data': {'model': 'EnhancedReceptionSimpleRNN', 'input_size': self.rec.input_size, 'reception_size': self.reception_size,
                                     'output_size': self.lin.out_features, 'unit_type': self.rec._get_name(),
                                     'num_layers': self.rec.num_layers, 'hidden_size': self.rec.hidden_size,
                                     'bias_fl': self.bias_fl}}

        if self.save_state:
            model_state = self.state_dict()
            for each in model_state:
                model_state[each] = model_state[each].cpu().data.numpy().tolist()
            model_data['state_dict'] = model_state

        miscfuncs.json_save(model_data, file_name, direc)

    # train_epoch runs one epoch of training
    def train_epoch(self, input_data, target_data, loss_fcn, optim, bs, init_len=200, up_fr=1000):
        # shuffle the segments at the start of the epoch
        shuffle = torch.randperm(input_data.shape[1])

        # Iterate over the batches
        ep_loss = 0
        for batch_i in range(math.ceil(shuffle.shape[0] / bs)):
            # Load batch of shuffled segments
            input_batch = input_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]
            target_batch = target_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]

            # Initialise network hidden state by processing some samples then zero the gradient buffers
            self(input_batch[0:init_len, :, :])
            self.zero_grad()

            # Choose the starting index for processing the rest of the batch sequence, in chunks of args.up_fr
            start_i = init_len
            batch_loss = 0
            # Iterate over the remaining samples in the mini batch
            for k in range(math.ceil((input_batch.shape[0] - init_len) / up_fr)):
                # Process input batch with neural network
                output = self(input_batch[start_i:start_i + up_fr, :, :])

                # Calculate loss and update network parameters
                loss = loss_fcn(output, target_batch[start_i:start_i + up_fr, :, :])
                loss.backward()
                optim.step()

                # Set the network hidden state, to detach it from the computation graph
                self.detach_hidden()
                self.zero_grad()

                # Update the start index for the next iteration and add the loss to the batch_loss total
                start_i += up_fr
                batch_loss += loss

            # Add the average batch loss to the epoch loss and reset the hidden states to zeros
            ep_loss += batch_loss / (k + 1)
            self.reset_hidden()
        return ep_loss / (batch_i + 1)

    # Only proc processes a the input data and calculates the loss, optionally grad can be tracked or not
    def process_data(self, input_data, target_data, loss_fcn, chunk, grad=False):
        with (torch.no_grad() if not grad else nullcontext()):
            output = torch.empty_like(target_data)
            for l in range(int(output.size()[0] / chunk)):
                output[l * chunk:(l + 1) * chunk] = self(input_data[l * chunk:(l + 1) * chunk])
                self.detach_hidden()
            # If the data set doesn't divide evenly into the chunk length, process the remainder
            if not (output.size()[0] / chunk).is_integer():
                output[(l + 1) * chunk:-1] = self(input_data[(l + 1) * chunk:-1])
            self.reset_hidden()
            loss = loss_fcn(output, target_data)
        return output, loss


class ConvSimpleRNN(nn.Module):
    """
    A simple ConvSimpleRNN class that consists of multiple Conv1d layers each one applying a series of dilated convolutions, with the dilation of each successive layer
    increasing by a factor of 'dilation_growth' followed by a single recurrent unit of type LSTM, GRU or Elman, followed by a fully connected layer
    """
    def __init__(self, input_size=1, dilation_num=6, dilation_growth=2, channels=6, kernel_size=3, output_size=1, unit_type="GRU", hidden_size=12, skip=0, bias_fl=True,
                 num_layers=1):
        super(ConvSimpleRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # Convolutional block
        self.dilation_num = dilation_num
        self.dilations = [dilation_growth ** layer for layer in range(dilation_num)]
        self.dilation_growth = dilation_growth
        self.kernel_size = kernel_size
        self.channels = channels
        self.conv = nn.ModuleList()
        dil_cnt = 0
        for dil in self.dilations:
            self.conv.append(nn.Conv1d(1 if dil_cnt == 0 else channels, out_channels=channels, kernel_size=kernel_size, dilation=dil, stride=1, padding=0, bias=True))
            dil_cnt = dil_cnt + 1
        # Recurrent block
        input_size=self.channels
        self.rec = wrapperargs(getattr(nn, unit_type), [input_size, hidden_size, num_layers])
        # Linear output, single neuron
        self.lin = nn.Linear(hidden_size, output_size, bias=bias_fl)
        self.bias_fl = bias_fl
        self.skip = skip
        self.save_state = True
        self.hidden = None

    def forward_conv(self, x):
        x = x.permute(1, 2, 0)
        y = x
        for n, layer in enumerate(self.conv):
            y = layer(y)
            y = torch.cat((torch.zeros(x.shape[0], self.channels, x.shape[2] - y.shape[2]), y), dim=2)
        return y.permute(2, 0, 1)

    def forward(self, x, hidden=None):
        x = self.forward_conv(x)
        if self.skip > 0:
            # save the residual for the skip connection
            res = x[:, :, 0:self.skip]
            if(hidden):
                x, self.hidden = self.rec(x, hidden)
                return ((self.lin(x) + res), self.hidden)
            else:
                x, self.hidden = self.rec(x, self.hidden)
                return self.lin(x) + res
        else:
            if(hidden):
                x, self.hidden = self.rec(x, hidden)
                return (self.lin(x), self.hidden)
            else:
                x, self.hidden = self.rec(x, self.hidden)
                return self.lin(x)

    # detach hidden state, this resets gradient tracking on the hidden state
    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()

    # changes the hidden state to None, causing pytorch to create an all-zero hidden state when the rec unit is called
    def reset_hidden(self):
        self.hidden = None

    # This functions saves the model and all its paraemters to a json file, so it can be loaded by a JUCE plugin
    def save_model(self, file_name, direc=''):
        if direc:
            miscfuncs.dir_check(direc)
        model_data = {'model_data': {'model': 'ConvSimpleRNN', 'input_size': self.input_size, 'skip': self.skip,
                                     'dilation_num': self.dilation_num, 'dilation_growth': self.dilation_growth,
                                     'channels': self.channels, 'kernel_size': self.kernel_size,
                                     'output_size': self.lin.out_features, 'unit_type': self.rec._get_name(),
                                     'num_layers': self.rec.num_layers, 'hidden_size': self.rec.hidden_size,
                                     'bias_fl': self.bias_fl}}

        if self.save_state:
            model_state = self.state_dict()
            for each in model_state:
                model_state[each] = model_state[each].cpu().data.numpy().tolist()
            model_data['state_dict'] = model_state

        miscfuncs.json_save(model_data, file_name, direc)

    # train_epoch runs one epoch of training
    def train_epoch(self, input_data, target_data, loss_fcn, optim, bs, init_len=200, up_fr=1000):
        # shuffle the segments at the start of the epoch
        shuffle = torch.randperm(input_data.shape[1])

        # Iterate over the batches
        ep_loss = 0
        for batch_i in range(math.ceil(shuffle.shape[0] / bs)):
            # Load batch of shuffled segments
            input_batch = input_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]
            target_batch = target_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]

            # Initialise network hidden state by processing some samples then zero the gradient buffers
            self(input_batch[0:init_len, :, :])
            self.zero_grad()

            # Choose the starting index for processing the rest of the batch sequence, in chunks of args.up_fr
            start_i = init_len
            batch_loss = 0
            # Iterate over the remaining samples in the mini batch
            for k in range(math.ceil((input_batch.shape[0] - init_len) / up_fr)):
                # Process input batch with neural network
                output = self(input_batch[start_i:start_i + up_fr, :, :])

                # Calculate loss and update network parameters
                loss = loss_fcn(output, target_batch[start_i:start_i + up_fr, :, :])
                loss.backward()
                optim.step()

                # Set the network hidden state, to detach it from the computation graph
                self.detach_hidden()
                self.zero_grad()

                # Update the start index for the next iteration and add the loss to the batch_loss total
                start_i += up_fr
                batch_loss += loss

            # Add the average batch loss to the epoch loss and reset the hidden states to zeros
            ep_loss += batch_loss / (k + 1)
            self.reset_hidden()
        return ep_loss / (batch_i + 1)

    # Only proc processes a the input data and calculates the loss, optionally grad can be tracked or not
    def process_data(self, input_data, target_data, loss_fcn, chunk, grad=False):
        with (torch.no_grad() if not grad else nullcontext()):
            output = torch.empty_like(target_data)
            for l in range(int(output.size()[0] / chunk)):
                output[l * chunk:(l + 1) * chunk] = self(input_data[l * chunk:(l + 1) * chunk])
                self.detach_hidden()
            # If the data set doesn't divide evenly into the chunk length, process the remainder
            if not (output.size()[0] / chunk).is_integer():
                output[(l + 1) * chunk:-1] = self(input_data[(l + 1) * chunk:-1])
            self.reset_hidden()
            loss = loss_fcn(output, target_data)
        return output, loss

def load_model(model_data):
    model_types = {'AdvancedClipSimpleRNN': AdvancedClipSimpleRNN, 'AsymmetricAdvancedClipSimpleRNN': AsymmetricAdvancedClipSimpleRNN, 'ConvSimpleRNN': ConvSimpleRNN}

    model_meta = model_data.pop('model_data')

    if model_meta['model'] == 'AdvancedClipSimpleRNN' or model_meta['model'] == 'AsymmetricAdvancedClipSimpleRNN':
        network = wrapperkwargs(model_types[model_meta.pop('model')], model_meta)
        if 'state_dict' in model_data:
            state_dict = network.state_dict()
            for each in model_data['state_dict']:
                state_dict[each] = torch.tensor(model_data['state_dict'][each])
            network.load_state_dict(state_dict)

    elif model_meta['model'] == 'ConvSimpleRNN':
        network = wrapperkwargs(model_types[model_meta.pop('model')], model_meta)
        if 'state_dict' in model_data:
            state_dict = network.state_dict()
            for each in model_data['state_dict']:
                state_dict[each] = torch.tensor(model_data['state_dict'][each])
            network.load_state_dict(state_dict)

    return network