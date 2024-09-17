#!/usr/bin/env python3

'''A set of PyTorch-based classes to build LRNN systems.

These classes include new autograd functions, new layers, new loss functions,
and new neural networks.
The new autograd functions include a gradient masking function that allows
tensors to require gradients only for some of their elements.
The new layers include a 2D locally connected recurrent layer, a 2D circular
kernel max pooling layer, a temporal max pooling layer, a Gaussian noise layer,
a temporal window layer, a downsampling layer, an upsampling layer with
coherent indices, a multiple-input upsampling layer, a fully connected
recurrent layer with sigmoid activation function, a rectified Gaussian mixture
distribution layer, a rectified Gaussian mixture peak layer, a concatenating
layer, a slicing layer, a triming layer, a non-linearity emulating adaptation
layer, and a generic function layer.
The new loss functions include an MSE loss (divided by the batch size), an L1
loss (divided by the batch size), a rectified Gaussian mixture negative log
likelihood loss, a weight decay loss affecting only the weights, a sparsity
loss, a sparsity loss for normalized data, and a slowness-oriented contrastive
loss.
The new neural networks include the neocortex-inspired locally recurrent neural
network (NILRNN), the locally recurrent block (LRB), the deep version of the
LRB, a shallow Gaussian mixture density network (MDN), a shallow MDN-based LRB
predictor, a GRU- and MDN-based LRB predictor, an upsampling neural network, a
shallow vanilla recurrent network, an upsampling recurrent neural network, and
a module to stack neural networks in different configurations that allows to
build hierarchical and U-shaped locally recurrent neural networks (HLRNNs and
U-LRNNs).
'''

import torch
from torch import nn
import torch.nn.functional as F
import math
import copy
from collections import OrderedDict
import typing
from typing import Any, List, Tuple, Dict, Callable
from typing_extensions import Self



def _get_circ_kernel(kernel_numel: int) -> Tuple[torch.Tensor, int]:
    '''Returns a circular kernel of a number of elements close to kernel_numel.

    Arguments:
        kernel_numel: The desired number of kernel elements.

    Returns:
        kernel: The kernel.
        kernel_numel: The actual number of kernel elements.
    '''
    var = -kernel_numel/(2*math.pi*math.log(0.1))
    rad = math.floor(math.sqrt(kernel_numel/math.pi))
    grid_y, grid_x = torch.meshgrid(torch.arange(-rad, rad + 1), torch.arange(
        -rad, rad + 1), indexing='ij')
    kernel = torch.exp(-(torch.square(grid_x) + torch.square(grid_y)) / (2 *
        var))
    kernel[kernel < 0.1] = 0
    kernel_numel = kernel.count_nonzero().item()
    kernel *= kernel_numel / kernel.sum()
    return kernel, kernel_numel


class _Rec(nn.Module):
    def __init__(self, nonlinearity: str):
        super(_Rec, self).__init__()
        if nonlinearity == 'sigm':
            self._act_fn = torch.sigmoid
        elif nonlinearity == 'tanh':
            self._act_fn = torch.tanh
        elif nonlinearity == 'relu':
            self._act_fn = torch.relu
        else:
            raise ValueError(f'{nonlinearity} is not a valid non-linearity')

    def forward(self, x: torch.Tensor, pred_sample: torch.Tensor, weight:
        torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        pred_list = []
        for x_sample in x:
            pred_sample = self._act_fn((weight @ torch.cat((x_sample,
                pred_sample)) + bias))
            pred_list.append(pred_sample)
        return torch.stack(pred_list)

_sigmrec_forward_fn = _Rec('sigm')#torch.jit.script(_Rec('sigm'))
_tanhrec_forward_fn = _Rec('tanh')#torch.jit.script(_Rec('tanh'))
_relurec_forward_fn = _Rec('relu')#torch.jit.script(_Rec('relu'))


class GradMaskFunction(torch.autograd.Function):
    '''An autograd function that sets to 0 the indicated gradients.'''
    @staticmethod
    def forward(ctx, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''Runs the forward pass (does nothing).

        Arguments:
            input: The input to the function.
            mask: The gradient mask.

        Returns:
            output: The output to the function (same as the input).
        '''
        ctx.set_materialize_grads(False)
        ctx.mask = mask
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        '''Runs the backward pass (masks the gradient).

        Arguments:
            grad_output: The gradient at the output of the function.

        Returns:
            grad_input: The gradient at the input of the function (the masked
                gradient).
            grad_mask: We are not interested in the mask gradient (None).
        '''
        if grad_output is None or not ctx.needs_input_grad[0]:
            return None, None
        return grad_output * ctx.mask, None

gradmask = GradMaskFunction.apply


class LocallyRec2d(nn.Module):
    '''A 2D locally recurrent layer.

    Attributes:
        in_features: An integer with the size of each input sample.
        out_features: An integer with the size of each output sample.
        out_shape: A tuple of integers with the shape of each output sample.
        kernel_numel: An integer with the number of elements of the recurrent
            connections window.
        nonlinearity: A string with the non-linearity to use.
        kernel_shape: A string with the shape of the recurrent kernel.
        kernel_side: If the kernel is square, an integer with the side of the
            recurrent kernel.
        weight: A 2D tensor of floats with the learnable weights of the module
            of shape (out_features, in_features + out_featues). The values are
            initialized according to Xavier initialization.
        bias: A 1D tensor of floats with the learnable bias of the module of
            shape (out_features). The values are initialized to 0.
        init_mask: A 2D tensor of floats defining the initialization gains of
            the connections in the recurrent part of the weight matrix.
        mask: A 2D tensor of floats defining the existing connections in the
            recurrent part of the weight matrix.
        _rec_forward_fn: A function that performs a forward pass over a
            recurrent layer with the weights and biases passed as parameter.
    '''
    def __init__(self, in_features: int, out_shape: Tuple[int], kernel_numel:
        int, nonlinearity: str = 'sigm', kernel_shape: str = 'circular') -> \
        None:
        '''Inits LocallyRec2d.

        Arguments:
            in_features: The size of each input sample.
            out_shape: The shape of each output sample.
            kernel_numel: The desired number of elements of the recurrent
                connections window.
            nonlinearity: The non-linearity to use.
            kernel_shape: The shape of the recurrent kernel.
        '''
        super(LocallyRec2d, self).__init__()
        self.in_features, self.out_shape = in_features, tuple(out_shape)
        self.nonlinearity = nonlinearity
        self.kernel_shape = kernel_shape
        self.out_features = out_shape[0] * out_shape[1]
        if kernel_shape == 'circular':
            kernel, self.kernel_numel = _get_circ_kernel(kernel_numel)
        elif kernel_shape == 'square':
            self.kernel_side = round((math.sqrt(kernel_numel) - 1) / 2) * 2 + 1
            self.kernel_numel = self.kernel_side ** 2
            kernel = torch.ones(self.kernel_side, self.kernel_side)
        else:
            raise ValueError(f'{kernel_shape} is not a valid kernel shape')
        self.init_mask = F.conv2d(torch.eye(self.out_features).reshape((
            self.out_features, 1) + self.out_shape), kernel.unsqueeze(
            0).unsqueeze(0), padding='same').reshape(self.out_features,
            self.out_features)
        self.mask = self.init_mask.clone()
        self.mask[self.mask > 0] = 1
        self.weight = nn.Parameter(torch.empty(self.out_features, in_features +
            self.out_features))
        self.bias = nn.Parameter(torch.empty(self.out_features))
        if nonlinearity == 'sigm':
            self._rec_forward_fn = _sigmrec_forward_fn
        elif nonlinearity == 'tanh':
            self._rec_forward_fn = _tanhrec_forward_fn
        elif nonlinearity == 'relu':
            self._rec_forward_fn = _relurec_forward_fn
        else:
            raise ValueError(f'{nonlinearity} is not a valid non-linearity')
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Initializes weight following Xavier initialization and bias to 0.'''
        bound = math.sqrt(6 / (self.in_features + self.kernel_numel +
            self.out_features))
        nn.init.uniform_(self.weight, -bound, bound)
        with torch.no_grad():
            self.weight[:,self.in_features:] *= self.mask
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None) -> \
        torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.
            h_0: The initial hidden state. Defaults to zeros if not provided.

        Returns:
            pred: The output batch (or prediction).
        '''
        if h_0 is None:
            h_0 = torch.zeros(self.out_features)
        weight = torch.cat((self.weight[:, :self.in_features],
            gradmask(self.weight[:, self.in_features:], self.mask)), 1)
        return self._rec_forward_fn(x, h_0, weight, self.bias)


class CircMaxPool2d(nn.Module):
    '''A 2D max pooling layer with circular-shaped kernel.

    Attributes:
        in_features: An integer with the size of each input sample.
        in_shape: A tuple of integers with the shape of each input sample.
        out_features: An integer with the size of each output sample.
        out_shape: A tuple of integers with the shape of each output sample.
        kernel_numel: An integer with the number of elements of the window to
            take a max over.
        stride: An integer with the stride of the window.
        mask: A 2D tensor of booleans defining the existing max pooling
            connections.
    '''
    def __init__(self, in_shape: Tuple[int], kernel_numel: int, stride: int) \
        -> None:
        ''' Inits CircMaxPool2d.

        Arguments:
            in_shape: The shape of each input sample.
            kernel_numel: The desired number of elements of the window to take
                a max over.
            stride: The stride of the window.
        '''
        super(CircMaxPool2d, self).__init__()
        self.in_shape, self.stride = in_shape, stride
        self.in_features = in_shape[0] * in_shape[1]
        kernel, self.kernel_numel = _get_circ_kernel(kernel_numel)
        mask = F.conv2d(torch.eye(self.in_features).reshape((self.in_features,
            1) + self.in_shape), kernel.unsqueeze(0).unsqueeze(0),
            padding='same').reshape(self.in_shape + self.in_shape)
        mask = mask[::stride,::stride,:,:]
        self.out_shape = tuple(mask.shape[0:2])
        self.out_features = self.out_shape[0] * self.out_shape[1]
        self.mask = mask.reshape(self.out_features, self.in_features) != 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.

        Returns:
            pred: The output batch (or prediction).
        '''
        pred_list = [x[:, mask_row].max(1).values for mask_row in self.mask]
        return torch.stack(pred_list, 1)


class TempMaxPool(nn.Module):
    '''A downsampling temporal max pooling layer.

    Attributes:
        kernel_size: An integer with the size of the window to take a max over.
        stride: An integer with the stride of the window.
        centered: A boolean indicating whether the kernel is centered or
            applied only over current and past samples.
        scale_factor: A float with the multiplier for the batch size.
        _pad_left: An integer with the left padding required for the input
            batch.
        _pad_right: An integer with the right padding required for the input
            batch.
    '''
    def __init__(self, kernel_size: int, stride: int, centered: bool = False) \
        -> None:
        '''Inits TempMaxPool.

        Arguments:
            kernel_size: The size of the window to take a max over.
            stride: The stride of the window.
            centered: The indicator of whether the kernel is centered or
            applied only over current and past samples.
        '''
        super(TempMaxPool, self).__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.centered = centered
        self.scale_factor = 1 / stride
        if centered:
            self._pad_left = math.floor(kernel_size / 2)
            self._pad_right = kernel_size - self._pad_left - 1
        else:
            self._pad_left = kernel_size - 1
            self._pad_right = 0

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None, *, shift: int
        = 0) -> torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.
            h_0: The initial hidden state. Defaults to -infs if not provided.
            shift: The sample at which the downsampling starts.

        Returns:
            pred: The output batch (or prediction).
            h_n: The final hidden state.
        '''
        if h_0 == None:
            h_0 = -torch.inf * torch.ones(self._pad_left, *x.shape[1:])
        leftpadded = torch.cat([h_0, x], 0)
        return torch.cat([leftpadded[shift:], -torch.inf * torch.ones(
            self._pad_right + shift, *x.shape[1:])],
            0).contiguous().as_strided((math.ceil(x.shape[0] / self.stride),
            self.kernel_size, x.shape[1]), (self.stride * x.shape[1],
            x.shape[1], 1)).max(1).values, leftpadded[leftpadded.shape[0] -
            self._pad_left:]


class Noise(nn.Module):
    '''A layer that adds in training mode Gaussian noise to its input.

    Attributes:
        std: A float with the standard deviation of the Gaussian noise added.
    '''
    def __init__(self, std: float) -> None:
        '''Inits Noise.

        Arguments:
            std: The standard deviation of the Gaussian noise added.
        '''
        super(Noise, self).__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.

        Returns:
            pred: The output batch (or prediction).
        '''
        if self.training:
            return x + self.std * torch.randn_like(x)
        else:
            return x


class TempWindow(nn.Module):
    '''A temporal window layer.

    Attributes:
        window_size: An integer with the size of the temporal window.
    '''
    def __init__(self, window_size: int) -> None:
        '''Inits TempWindow.

        Arguments:
            window_size: The size of the temporal window.
        '''
        super(TempWindow, self).__init__()
        self.window_size = window_size

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None) -> \
        torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.
            h_0: The initial hidden state. Defaults to the first sample if not
                provided.

        Returns:
            pred: The output batch (or prediction).
            h_n: The final hidden state.
        '''
        if h_0 is None:
            h_0 = x[0].expand(self.window_size - 1, *x.shape[1:])
        padded = torch.cat([h_0, x], 0)
        return padded.contiguous().as_strided((x.shape[0], self.window_size *
            x.shape[1]), (x.shape[1], 1)), padded[1 - self.window_size:]


class Downsample(nn.Module):
    '''A downsampling layer by an integer factor.

    Attributes:
        stride: An integer with the downsampling factor.
        scale_factor: A float with the multiplier for the batch size.
    '''
    def __init__(self, stride: int) -> None:
        '''Inits Downsample.

        Arguments:
            stride: The stride of the window.
        '''
        super(Downsample, self).__init__()
        self.stride = stride
        self.scale_factor = 1 / stride

    def forward(self, x: torch.Tensor, *, shift: int = 0) -> torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.
            shift: The sample at which the downsampling starts.

        Returns:
            pred: The output batch (or prediction).
        '''
        return x[shift::self.stride]


class Upsample(nn.Module):
    '''An interpolating layer that keeps indices coherent.

    Attributes:
        scale_factor: An integer with the multiplier for the batch size.
        mode: A string with the algorithm used for upsampling.
    '''
    def __init__(self, scale_factor: int, mode: str = 'constant') -> None:
        '''Inits Upsample.

        Arguments:
            scale_factor: The multiplier for the batch size.
            mode: The algorithm used for upsampling.
        '''
        super(Upsample, self).__init__()
        self.scale_factor, self.mode = scale_factor, mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.

        Returns:
            pred: The output batch (or prediction).
        '''
        if self.mode == 'constant':
            return x.repeat_interleave(self.scale_factor, 0)
        return F.pad(F.interpolate(x.T.unsqueeze(0), size=(x.shape[0] - 1) *
            self.scale_factor + 1, mode=self.mode, align_corners=True).squeeze(
            0), (0, self.scale_factor-1), mode='replicate').T


class MultiUpsample(nn.Module):
    '''An interpolating layer over multiple inputs that keeps indices coherent.

    Attributes:
        scale_factor: An integer with the multiplier for the batch size.
        mode: A string with the algorithm used for upsampling.
    '''
    def __init__(self, scale_factor: int, mode: str = 'constant') -> None:
        '''Inits MultiUpsample.

        Arguments:
            scale_factor: The multiplier for the batch size.
            mode: The algorithm used for upsampling.
        '''
        super(MultiUpsample, self).__init__()
        self.scale_factor, self.mode = scale_factor, mode

    def forward(self, *xs: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Performs a forward pass.

        Arguments:
            xs: The input batches.

        Returns:
            preds: The output batches (or predictions).
        '''
        if self.mode == 'constant':
            return tuple(x.repeat_interleave(self.scale_factor, 0) for x in xs)
        return tuple(F.pad(F.interpolate(x.T.unsqueeze(0), size=(x.shape[0] -
            1) * self.scale_factor + 1, mode=self.mode, align_corners=
            True).squeeze(0), (0, self.scale_factor-1), mode='replicate').T for
            x in xs)


class SigmRec(nn.Module):
    '''A vanilla recurrent layer with sigmoid activation function.

    Attributes:
        in_features: An integer with the size of each input sample.
        out_features: An integer with the size of each output sample.
        weight: A 2D tensor of floats with the learnable weights of the module
            of shape (out_features, in_features + out_features). The values are
            initialized according to Xavier initialization.
        bias: A 1D tensor of floats with the learnable bias of the module of
            shape (out_features). The values are initialized to 0.
    '''
    def __init__(self, in_features: int, out_features: int) -> None:
        ''' Inits SigmRec.

        Arguments:
            in_features: The size of each input sample.
            out_features: The size of each output sample.
        '''
        super(SigmRec, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features +
            out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Initializes weight following Xavier initialization and bias to 0.'''
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None) -> \
        torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.
            h_0: The initial hidden state. Defaults to zeros if not provided.

        Returns:
            pred: The output batch (or prediction).
        '''
        if h_0 is None:
            h_0 = torch.zeros(self.out_features)
        return _sigmrec_forward_fn(x, h_0, self.weight, self.bias)


class RectGaussMix(nn.Module):
    '''A layer that samples a rectified Gaussian mixture distribution.'''
    def __init__(self) -> None:
        ''' Inits RectGaussMix.'''
        super(RectGaussMix, self).__init__()

    def forward(self, mean: torch.Tensor, var: torch.Tensor, coef:
        torch.Tensor) -> torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            mean: The input mean batch.
            var: The input variance batch.
            coef: The input component mixing coefficient batch.

        Returns:
            pred: The output batch (or prediction).
        '''
        comp = torch.multinomial(coef, 1).flatten()
        pred = torch.normal(mean[range(mean.shape[0]), comp], torch.sqrt(var[
            range(var.shape[0]), comp]))
        pred[pred < 0] = 0
        return pred


class RectGaussMixPeak(nn.Module):
    '''A layer that finds a rectified Gaussian mixture distribution peak.'''
    def __init__(self) -> None:
        ''' Inits RectGaussMixPeak.'''
        super(RectGaussMixPeak, self).__init__()

    def forward(self, mean: torch.Tensor, var: torch.Tensor, coef:
        torch.Tensor) -> torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            mean: The input mean batch.
            var: The input variance batch (not used).
            coef: The input component mixing coefficient batch.

        Returns:
            pred: The output batch (or prediction).
        '''
        comp = coef.argmax(-1)
        pred = mean[range(mean.shape[0]), comp]
        pred[pred < 0] = 0
        return pred


class Cat(nn.Module):
    '''A layer that concatenates the given tensors in the given dimension.

    Attributes:
        dim: An integer with the dimension over which the tensors are
            concatenated.
    '''
    def __init__(self, dim: int = -1) -> None:
        '''Inits Cat.

        Arguments:
            dim: The dimension over which the tensors are concatenated.
        '''
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            xs: The input batches.

        Returns:
            pred: The output batch (or prediction).
        '''
        return torch.cat(xs, self.dim)


class Slice(nn.Module):
    '''A layer that outputs a specific slice of its inputs.

    Atributes:
        start: An integer with the starting index where the slicing of input
            starts.
        stop: An integer with the ending index where the slicing of input
            stops.
        dim: An integer with the dimension over which the slicing is performed.
    '''
    def __init__(self, start: int, stop: int, dim: int = -1) -> None:
        '''Inits Slice.

        Arguments:
            start: The starting index where the slicing of input starts.
            stop: The ending index where the slicing of input stops.
            dim: The dimension over which the slicing is performed.
        '''
        super(Slice, self).__init__()
        self.start, self.stop = start, stop
        self.dim = dim
        self._length = stop - start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.

        Returns:
            pred: The output batch (or prediction).
        '''
        return x.narrow(self.dim, self.start, self._length)


class TrimLike(nn.Module):
    '''A layer that trims the end of its input(s) to the size of its first one.

    Attributes:
        dim: An integer with the dimension over which the trimming is
            performed.
    '''
    def __init__(self, dim: int = 0) -> None:
        '''Inits TrimLike.

        Arguments:
            dim: The dimension over which the trimming is performed.
        '''
        super(TrimLike, self).__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor, *targets: torch.Tensor) -> Tuple[
        torch.Tensor]:
        '''Performs a forward pass.

        Arguments:
            input: The input reference batch.
            targets: The input batches to trim.

        Returns:
            preds: The output batches (or predictions).
        '''
        return tuple(x.narrow(self.dim, 0, input.shape[self.dim]) for x in
            targets)


class EmulateNonlinearity(nn.Module):
    '''A layer that emulates a specific non-linearity at the previous layer.

    Attributes:
        actual: A string with the non-linearity at the previous layer.
        desired: A string with the non-linearity to emulate.
        _emulate_fn: A function that emulates the desired non-linearity.
    '''
    def __init__(self, actual: str, desired: str) -> None:
        '''Inits EmulateNonlinearity.

        Arguments:
            actual: The non-linearity at the previous layer.
            desired: The non-linearity to emulate.
        '''
        super(EmulateNonlinearity, self).__init__()
        self.actual, self.desired = actual, desired
        if actual == desired:
            self._emulate_fn = lambda x: x
        elif actual == 'sigm' and desired == 'tanh':
            self._emulate_fn = lambda x: 2 * x - 1
        elif actual == 'sigm' and desired == 'relu':
            self._emulate_fn = lambda x: x
        elif actual == 'tanh' and desired == 'sigm':   
            self._emulate_fn = lambda x: (x + 1) / 2
        elif actual == 'tanh' and desired == 'relu':
            self._emulate_fn = lambda x: (x + 1) / 2
        elif actual == 'relu' and desired == 'sigm':
            self._emulate_fn = lambda x: (x + .1).tanh()
        elif actual == 'relu' and desired == 'tanh':
            self._emulate_fn = lambda x: 2 * (x + .1).tanh() - 1
        else:
            raise ValueError(f'{actual} or {desired} are not valid '
                             f'non-linearities')

    def  forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.

        Returns:
            pred: The output batch (or prediction).
        '''
        return self._emulate_fn(x)


class Function(nn.Module):
    '''A layer that implements a standard function.

    Atributes:
        function: A function that is called when the forward function is
            called.
    '''
    def __init__(self, function: Callable[[Any], Any]) -> None:
        '''Inits Function.

        Arguments:
            function: It is called when the forward function is called.
        '''
        super(Function, self).__init__()
        self.function = function

    def forward(self, *args: Tuple[Any]) -> Any:
        '''Performs a forward pass.

        Arguments:
            args: Any input(s).

        Returns:
            outputs: Any output(s).
        '''
        return self.function(*args)


class MSELoss(nn.Module):
    '''An MSE loss term that divides the sum of squares by the batch size.

    Attributes:
        get_weight_fn: A function that returns an ND tensor of floats with the
            weights that multiply the squares.
    '''
    def __init__(self, get_weight_fn: Callable[[int], torch.Tensor] = None) \
        -> None:
        '''Inits MSELoss.

        Arguments:
            get_weight_fn: A function that returns the weights that multiply
                the squares.
        '''
        super(MSELoss, self).__init__()
        self.get_weight_fn = get_weight_fn
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> \
        torch.Tensor:
        '''Returns the loss.

        Arguments:
            input: The hypothesized value.
            target: The actual value.

        Returns:
            loss: The resultant loss.
        '''
        if self.get_weight_fn is None:
            return (torch.square(input - target)).sum() / (2 * input.shape[0])
        return (self.get_weight_fn(input.shape[0]) * torch.square(input -
            target)).sum() / (2 * input.shape[0])


class L1Loss(nn.Module):
    '''An L1 loss term that divides the sum of absolutes by the batch size.

    Attributes:
        get_weight_fn: A function that returns an ND tensor of floats with the
            weights that multiply the absolutes.
    '''
    def __init__(self, get_weight_fn: Callable[[int], torch.Tensor] = None) \
        -> None:
        '''Inits L1Loss.

        Arguments:
            get_weight_fn: A function that returns the weights that multiply
                the absolutes.
        '''
        super(L1Loss, self).__init__()
        self.get_weight_fn = get_weight_fn
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> \
        torch.Tensor:
        '''Returns the loss.

        Arguments:
            input: The hypothesized value.
            target: The actual value.

        Returns:
            loss: The resultant loss.
        '''
        if self.get_weight_fn is None:
            return (torch.abs(input - target)).sum() / input.shape[0]
        return (self.get_weight_fn(input.shape[0]) * torch.abs(input -
            target)).sum() / input.shape[0]


class RectGaussMixNLLLoss(nn.Module):
    '''A negative log likelihood loss for mixtures of rectified Gaussians.

    Attributes:
        get_weight_fn: A function that returns an ND tensor of floats with the
            weights that multiply the negative log-likelihoods.
    '''
    def __init__(self, get_weight_fn: Callable[[int], torch.Tensor] = None) \
        -> None:
        '''Inits RectGaussMixNLLLoss.

        Arguments:
            get_weight_fn: A function that returns the weights that multiply
                the absolutes.
        '''
        super(RectGaussMixNLLLoss, self).__init__()
        self.get_weight_fn = get_weight_fn

    def forward(self, mean: torch.Tensor, var: torch.Tensor, coef:
            torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''Returns the loss.

        Arguments:
            mean: The hypothesized means of the non-rectified Gaussians.
            var: The hypothesized variances of the non-rectified Gaussians.
            coef: The mixing coefficients (or porbabilities) weighting the
                rectified Gaussians.
            target: The actual value.

        Returns:
            loss: The resultant loss.
        '''
        target = target.unsqueeze(-2)
        disc = (target <= 0).expand_as(mean)
        cont = (target > 0).expand_as(mean)
        target = target.expand_as(mean)
        #ls = torch.empty_like(mean)
        #ls[disc] = torch.special.ndtr(-mean[disc] / torch.sqrt(var[disc]))
        #ls[cont] = torch.exp(-(target[cont] - mean[cont]) ** 2 / (2 * var[
        #    cont])) / torch.sqrt(2 * torch.pi * var[cont])
        #nll = -torch.log((coef * ls.prod(-1)).sum(-1))
        lls = torch.empty_like(mean)
        lls[disc] = torch.special.log_ndtr(-mean[disc] / torch.sqrt(var[disc]))
        lls[cont] = -(target[cont] - mean[cont]) ** 2 / (2 * var[cont]) - \
            torch.log(torch.sqrt(2 * torch.pi * var[cont]))
        nll = -(torch.log(coef) + lls.sum(-1)).logsumexp(-1)
        if self.get_weight_fn is None:
            return nll.sum() / target.shape[0]
        return (self.get_weight_fn(target.shape[0]) * nll).sum() / \
            target.shape[0]


class WeightDecayLoss(nn.Module):
    '''A weight decay loss term affecting only the weight matrices.

    Attributes:
        weights: A list of ND tensors of floats with the weights to be
            regularized.
    '''
    def __init__(self, model: nn.Module) -> None:
        '''Inits WeightDecayLoss.

        Arguments:
            model: The model with the weights to be regularized.
        '''
        super(WeightDecayLoss, self).__init__()
        weights = []
        for parameter in model.named_parameters():
            if 'weight' in parameter[0]:
                weights.append(parameter[1])
        self.weights = weights

    def forward(self) -> torch.Tensor:
        '''Returns the loss.

        Returns:
            loss: The resultant loss.
        '''
        return sum(torch.square(weight).sum() for weight in self.weights) / 2


def _sigm_sparsity_loss_fn(vble, rho):
    vble_rho = torch.clamp(vble, 1e-5, 1 - 1e-5).mean(0)
    return (rho * torch.log2(rho / vble_rho) + (1 - rho) * torch.log2((1 - rho)
        / (1 - vble_rho))).sum()

def _tanh_sparsity_loss_fn(vble, rho):
    vble_rho = (torch.clamp(vble, -1 + 1e-5, 1 - 1e-5).mean(0) + 1) / 2
    return (rho * torch.log2(rho / vble_rho) + (1 - rho) * torch.log2((1 - rho)
        / (1 - vble_rho))).sum()

def _relu_sparsity_loss_fn(vble, rho):
    return vble.abs().sum() / vble.shape[0]

class SparsityLoss(nn.Module):
    '''A sparsity loss term on the given list of variables.

    Attributes:
        rho: A float with the desired sparsity parameter.
        vbles: A list of ND tensors of floats with the sparse variables.
        nonlinearity: A string with the non-linearity at the layer(s).
        _loss_fn: A function that obtains the sparsity loss for one tensor.
    '''
    def __init__(self, rho: float, vbles: List[torch.Tensor], nonlinearity: str
        = 'sigm') -> None:
        '''Inits SparsityLoss.

        Arguments:
            rho: The desired sparsity parameter.
            vbles: The sparse variables.
            nonlinearity: The non-linearity at the layer(s).
        '''
        super(SparsityLoss, self).__init__()
        self.rho, self.vbles, self.nonlinearity = rho, vbles, nonlinearity
        if nonlinearity == 'sigm':
            self._loss_fn = _sigm_sparsity_loss_fn
        elif nonlinearity == 'tanh':
            self._loss_fn = _tanh_sparsity_loss_fn
        elif nonlinearity == 'relu':
            self._loss_fn = _relu_sparsity_loss_fn
        else:
            raise ValueError(f'{nonlinearity} is not a valid non-linearity')

    def forward(self) -> torch.Tensor:
        '''Returns the loss.

        Returns:
            loss: The resultant loss.
        '''
        losses = []
        for vble in self.vbles:
            losses.append(self._loss_fn(vble, self.rho))
        return sum(losses)


class NormSparsityLoss(nn.Module):
    '''A sparsity loss term on the given normalized list of variables.

    Attributes:
        vbles: A list of ND tensors of floats with the sparse variables.
    '''
    def __init__(self, vbles: List[torch.Tensor]) -> None:
        '''Inits NormSparsityLoss.

        Arguments:
            vbles: The sparse variables.
        '''
        super(NormSparsityLoss, self).__init__()
        self.vbles = vbles

    def forward(self) -> torch.Tensor:
        '''Returns the loss.

        Returns:
            loss: The resultant loss.
        '''
        losses = []
        for vble in self.vbles:
            vble = vble.reshape(-1, vble.shape[-1]).clone()
            vble[torch.arange(vble.shape[0]), vble.argmax(1)] = 0
            losses.append(vble.sum() / vble.shape[0])
        return sum(losses)


class SlownessLoss(nn.Module):
    '''A contrastive loss term based on the principle of slowness.

    Attributes:
        delta: An integer with the maximum shift in time on the representation
            for which the loss minimizes the change.
        _mse_loss_fn: A function that obtains the loss given the two shifted
            versions of the representation.
    '''
    def __init__(self, delta: int) -> None:
        '''Inits SlownessLoss.

        Arguments:
            delta: The maximum shift in time on the representation for which
                the loss minimizes the change.
        '''
        super(SlownessLoss, self).__init__()
        self.delta = delta
        self._mse_loss_fn = MSELoss()

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        '''Returns de loss.

        Arguments:
            feats: The representation batch.

        Returns:
            loss: The resultant loss.
        '''
        return sum(self._mse_loss_fn(feats[:-delta], feats[delta:]) for delta
            in range(1, self.delta + 1)) / self.delta


class NILRNN(nn.Module):
    '''The Neocortex-Inspired Locally Recurrent Neural Network.

    Attributes:
        in_features: An integer with the size of each input sample.
        rec_shape: A tuple of integers with the shape of the locally recurrent
            layer.
        out_features: An integer with the size of each output sample.
        out_shape: A tuple of integers with the shape of each output sample.
        train_out_features: An integer with the size of each training output
            sample.
        rec_kernel_numel: An integer with the number of elements of the
            recurrent connections window.
        pool_kernel_numel: An integer with the number of elements of the max
            pooling window.
        pool_stride: An integer with the stride of the max pooling window.
        train_out_channels: An integer with the number of channels in the
            training output.
        dropout_p: A float with the probability of an element in the dropout
            layer to be zero-ed.
        noise_std: A float with the standard deviation of the noise layer.
        rec_nonlinearity: A string with the non-linearity to use at the locally
            recurrent layer.
        train_out_nonlinearity: A string with the non-linearity to use at the
            training output layer.
        kernel_shape: A string with the shape of the recurrent and max pooling
            kernels.
        return_contr: A boolean indicating whether the forward function will
            return the output for the contrastive loss in training mode.
        sparse_vbles: A list of ND tensors of floats with the variables over
            which to apply the sparsity loss.
        sparse_nonlinearity: A string with the non-linearity to use at the
            sparse (locally recurrent) layer.
        dropout_1d: A PyTorch module with the dropout layer.
        noise: A PyTorch module with the noise layer.
        locally_rec_2d: A PyTorch module with the locally recurrent layer.
        max_pool_2d: A PyTorch module with the max pooling layer.
        decoder: A PyTorch module with the output layer of training mode.
        pool_kernel_side: If the kernel is square, an integer with the side of
            the max pooling kernel.
        get_max_pool_input: A function that prepares the input for the max
            pooling layer.
        _train_out_act_fn: An activation function used at the training output.
        _last_mask: A torch tensor with the mask with the output elements to
             be considered in the loss function.
    '''
    def __init__(self, in_features: int, rec_shape: Tuple[int],
        rec_kernel_numel: int, pool_kernel_numel: int, pool_stride: int,
        train_out_channels: int, dropout_p: float = 0, noise_std: float = 0,
        rec_nonlinearity: str = 'sigm', train_out_nonlinearity: str = 'sigm',
        kernel_shape: str = 'circular', return_contr: bool = False) -> None:
        '''Inits NILRNN.

        Arguments:
            in_features: The size of each input sample.
            rec_shape: The shape of the locally recurrent layer.
            rec_kernel_numel: The desired number of elements of the recurrent
                connections window.
            pool_kernel_numel: The desired number of elements of the max
                pooling window.
            pool_stride: The stride of the max pooling window.
            train_out_channels: The number of channels of the training output.
            dropout_p: The probability of an element in the dropout layer to be
                zero-ed.
            noise_std: The standard deviation of the noise layer.
            rec_nonlinearity: The non-linearity to use at the locally recurrent
                layer.
            train_out_nonlinearity: The non-linearity to use at the training
                output layer.
            kernel_shape: The shape of the recurrent and max pooling kernels.
            return_contr: If true, the forward function will return the output
                for the contrastive loss in training mode.
        '''
        super(NILRNN, self).__init__()
        self.in_features, self.rec_shape, self.train_out_channels = \
            in_features, rec_shape, train_out_channels
        self.pool_stride = pool_stride
        self.dropout_p, self.noise_std = dropout_p, noise_std
        self.rec_nonlinearity, self.train_out_nonlinearity = \
            rec_nonlinearity, train_out_nonlinearity
        self.kernel_shape = kernel_shape
        self.return_contr = return_contr
        self.train_out_features = in_features * train_out_channels
        self.sparse_vbles = []
        self.sparse_nonlinearity = rec_nonlinearity
        self._last_mask = torch.tensor([])
        self.dropout_1d = nn.Dropout1d(dropout_p)
        self.noise = Noise(noise_std)
        self.locally_rec_2d = LocallyRec2d(in_features, rec_shape,
            rec_kernel_numel, rec_nonlinearity, kernel_shape)
        self.rec_kernel_numel = self.locally_rec_2d.kernel_numel
        if kernel_shape == 'circular':
            self.get_max_pool_input = lambda a: a
            self.max_pool_2d = CircMaxPool2d(rec_shape, pool_kernel_numel,
                pool_stride)
            self.out_features = self.max_pool_2d.out_features
            self.out_shape = self.max_pool_2d.out_shape
            self.pool_kernel_numel = self.max_pool_2d.kernel_numel
        elif kernel_shape == 'square':
            self.pool_kernel_side = round((math.sqrt(pool_kernel_numel) - 1) /
                2) * 2 + 1
            self.pool_kernel_numel = self.pool_kernel_side ** 2
            self.get_max_pool_input = lambda a: a.reshape(-1, *rec_shape)
            self.max_pool_2d = nn.MaxPool2d(self.pool_kernel_side, pool_stride,
                round((self.pool_kernel_side - 1) / 2))
            self.out_shape = (math.ceil(rec_shape[1] / pool_stride), math.ceil(
                rec_shape[1] / pool_stride))
            self.out_features = self.out_shape[0] * self.out_shape[1]
        else:
            raise ValueError(f'{kernel_shape} is not a valid kernel shape')
        self.decoder = nn.Linear(math.prod(rec_shape), train_out_channels *
            in_features)
        if train_out_nonlinearity == 'sigm':
            self._train_out_act_fn = torch.sigmoid
        elif train_out_nonlinearity == 'tanh':
            self._train_out_act_fn = torch.tanh
        elif train_out_nonlinearity == 'relu':
            self._train_out_act_fn = torch.relu
        elif train_out_nonlinearity == 'none':
            self._train_out_act_fn = lambda a: a
        else:
            raise ValueError(f'{train_out_nonlinearity} is not a valid '
                             f'non-linearity')
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Initializes weight following Xavier initialization and bias to 0.'''
        self.locally_rec_2d.reset_parameters()
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.
            h_0: The initial hidden state. Defaults to zeros if not provided.

        Returns:
            pred: The output batch (or prediction).
            pred2: An additional output batch for contrastive loss if required.
            y: The desired output batch (only in training mode).
            h_n: The final hidden state.
        '''
        del self.sparse_vbles[:]
        a = self.locally_rec_2d(self.noise(self.dropout_1d(x.T).T), h_0)
        #if not a.isfinite().all():
        #    if not self.locally_rec_2d.weight.isfinite().all() and \
        #        self.decoder.weight.all():
        #        raise ValueError(
        #            'Locally recurrent layer weights diverged due to exploding'
        #            ' gradients. Consider lowering the learning rate')
        #    if self.locally_rec_2d.weight.isfinite().all() and \
        #        self.locally_rec_2d.bias.isfinite().all() and \
        #        x.isfinite().all() and h_0.isfinite().all():
        #        raise ValueError(
        #            'Batch diverged on forward propagation through the locally'
        #            ' recurrent layer probably due to very large weights. '
        #            'Consider lowering the learning rate')
        self.sparse_vbles.append(a)
        if self.training:
            pred = self._train_out_act_fn(self.decoder(a))
            y = F.pad(x.T, (0, self.train_out_channels - 1), mode=
                'replicate').T.contiguous().as_strided((x.shape[0],
                self.train_out_channels * self.in_features), (self.in_features,
                1))
            if self.return_contr:
                return pred, self.max_pool_2d(self.get_max_pool_input(
                    a)).reshape(-1, self.out_features), y, a[-1]
            else:
                return pred, y, a[-1]
        else:
            return self.max_pool_2d(self.get_max_pool_input(a)).reshape(-1,
                self.out_features), a[-1]

    def get_mask(self, batch_size: int) -> torch.Tensor:
        '''Returns a mask for the loss function.

        Arguments:
            batch_size: The size of the input batch.

        Returns:
            mask: The mask with the output elements to be considered in the
                loss function.
        '''
        if batch_size == self._last_mask.shape[0]:
            return self._last_mask
        mask = torch.ones(batch_size, self.train_out_channels *
            self.in_features)
        for i in range(1, min(self.train_out_channels, batch_size + 1)):
            mask[-i, i * self.in_features:] = 0
        self._last_mask = mask
        return mask


class LRB(nn.Module):
    '''The Locally Recurrent Block (NILRNN block + temporal max pooling layer).

    Attributes:
        in_features: An integer with the size of each input sample.
        out_features: An integer with the size of each output sample.
        out_shape: A tuple of integers with the shape of each output sample.
        train_out_features: An integer with the size of each training output
            sample.
        rec_shape: A tuple of integers with the shape of the locally recurrent
            layer.
        rec_kernel_numel: An integer with the number of elements of the
            recurrent connections window.
        pool_kernel_numel: An integer with the number of elements of the max
            pooling window.
        pool_stride: An integer with the stride of the max pooling window.
        temp_kernel_size: An integer with the size of the temporal max pooling
            window to take a max over.
        temp_stride: An integer with the stride of the temporal max pooling
            window.
        scale_factor: A float with the multiplier for the batch size.
        train_out_channels: An integer with the number of channels of the
            training output.
        dropout_p: A float with the probability of an element in the dropout
            layer to be zero-ed.
        noise_std: A float with the standard deviation of the noise layer.
        rec_nonlinearity: A string with the non-linearity to use at the locally
            recurrent layer.
        train_out_nonlinearity: A string with the non-linearity to use at the
            training output layer.
        kernel_shape: A string with the shape of the recurrent and max pooling
            kernels.
        temp_pos: A string with the position of the temporal max pooling layer
            with respect to the NILRNN.
        temp_kernel_centered: A boolean indicating whether the temporal max
            pooling kernel is centered or applied only over current and past
            samples.
        return_contr: A boolean indicating whether the forward function will
            return the output for the contrastive loss in training mode.
        sparse_vbles: A list of ND tensors of floats with the variables over
            which to apply the sparsity loss.
        sparse_nonlinearity: A string with the non-linearity to use at the
            sparse (locally recurrent) layer.
        nilrnn: A PyTorch module with the NILRNN.
        temp_max_pool: A PyTorch module with the temporal max pooling layer.
        _temp_pos: A boolean with the position of the temporal max pooling
            layer with respect to the NILRNN.
    '''
    def __init__(self, in_features: int, rec_shape: Tuple[int],
        rec_kernel_numel: int, pool_kernel_numel: int, pool_stride: int,
        temp_kernel_size: int, temp_stride: int, train_out_channels: int,
        dropout_p: float = 0, noise_std: float = 0, rec_nonlinearity: str =
        'relu', train_out_nonlinearity: str = 'relu', kernel_shape: str =
        'circular', temp_pos: str = 'first', temp_kernel_centered: bool =
        False, return_contr: bool = True) -> None:
        '''Inits LRB.

        Arguments:
            in_features: The size of each input sample.
            rec_shape: The shape of the locally recurrent layer.
            rec_kernel_numel: The desired number of elements of the recurrent
                connections window.
            pool_kernel_numel: The desired number of elements of the max
                pooling window.
            pool_stride: The stride of the max pooling window.
            temp_kernel_size: The size of the temporal max pooling window to
                take a max over.
            temp_stride: The stride of the temporal max pooling window.
            train_out_channels: The number of channels of the training output.
            dropout_p: The probability of an element in the dropout layer to be
                zero-ed.
            noise_std: The standard deviation of the noise layer.
            rec_nonlinearity: The non-linearity to use at the locally recurrent
                layer.
            train_out_nonlinearity: The non-linearity to use at the training
                output layer.
            kernel_shape: The shape of the recurrent and max pooling kernels.
            temp_pos: The position of the temporal max pooling layer with
                respect to the NILRNN.
            temp_kernel_centered: The indicator of whether the temporal max
                pooling kernel is centered or applied only over current and
                past samples.
            return_contr: If true, the forward function will return the output
                for the contrastive loss in training mode.
        '''
        super(LRB, self).__init__()
        self.in_features, self.rec_shape, self.train_out_channels = \
            in_features, rec_shape, train_out_channels
        self.pool_stride = pool_stride
        self.temp_kernel_size, self.temp_stride = temp_kernel_size, temp_stride
        self.dropout_p, self.noise_std = dropout_p, noise_std
        self.rec_nonlinearity, self.train_out_nonlinearity = \
            rec_nonlinearity, train_out_nonlinearity
        self.kernel_shape = kernel_shape
        self.temp_pos, self.temp_kernel_centered = temp_pos, \
            temp_kernel_centered
        self.return_contr = return_contr
        self.nilrnn = NILRNN(in_features, rec_shape, rec_kernel_numel,
            pool_kernel_numel, pool_stride, train_out_channels, dropout_p,
            noise_std, rec_nonlinearity, train_out_nonlinearity, kernel_shape,
            return_contr)
        self.out_features = self.nilrnn.out_features
        self.out_shape = self.nilrnn.out_shape
        self.train_out_features = self.nilrnn.train_out_features
        self.rec_kernel_numel = self.nilrnn.rec_kernel_numel
        self.pool_kernel_numel = self.nilrnn.pool_kernel_numel
        self.sparse_vbles = self.nilrnn.sparse_vbles
        self.sparse_nonlinearity = rec_nonlinearity
        self.temp_max_pool = TempMaxPool(temp_kernel_size, temp_stride,
            temp_kernel_centered)
        self.scale_factor = self.temp_max_pool.scale_factor
        if temp_pos == 'first':
            self._temp_pos = False
            self.train_scale_factor = self.scale_factor
        elif temp_pos == 'last':
            self._temp_pos = True
            self.train_scale_factor = 1
        else:
            raise ValueError(f'{temp_pos} is not a valid temporal max pooling '
                             f'layer position')

    def reset_parameters(self) -> None:
        '''Initializes weight following Xavier initialization and bias to 0.'''
        self.nilrnn.reset_parameters()

    def forward(self, x: torch.Tensor, rec_0: torch.Tensor = None, temp_0:
        torch.Tensor = None) -> Tuple[torch.Tensor]:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.
            rec_0: The initial locally recurrent layer hidden state. Defaults
                to zeros if not provided.
            temp_0: The initial temporal max pooling layer hidden state.
                Defaults to -infs if not provided.

        Returns:
            pred: The output batch (or prediction).
            pred2: An additional output batch for contrastive loss if required.
            y: The desired output batch (only in training mode).
            rec_n: The final locally recurrent layer hidden state.
            temp_n: The final temporal max pooling layer hidden state.
        '''
        if self._temp_pos:
            *output, rec_n = self.nilrnn(x, rec_0)
            if self.training:
                return tuple(output) + (rec_n, None)
            else:
                pred, temp_n = self.temp_max_pool(output[0], temp_0)
                return pred, rec_n, temp_n
        else:
            a, temp_n = self.temp_max_pool(x, temp_0)
            *output, rec_n = self.nilrnn(a, rec_0)
            return tuple(output) + (rec_n, temp_n)

    def get_mask(self, batch_size: int) -> torch.Tensor:
        '''Returns a mask for the loss function.

        Arguments:
            batch_size: The size of the input batch (after the temporal max
                pooling layer).

        Returns:
            mask: The mask with the output elements to be considered in the
                loss function.
        '''
        return self.nilrnn.get_mask(batch_size)


class DeepLRB(nn.Module):
    '''The deep variant of the LRB.

    Attributes:
        in_features: An integer with the size of each input sample.
        hidden_features: An integer with the size of the intermediate hidden
            layers.
        rec_shape: A tuple of integers with the shape of the locally recurrent
            layer.
        out_features: An integer with the size of each output sample.
        out_shape: A tuple of integers with the shape of each output sample.
        train_out_features: An integer with the size of each training output
            sample.
        scale_factor: A float with the multiplier for the batch size.
        rec_kernel_numel: An integer with the number of elements of the
            recurrent connections window.
        pool_kernel_numel: An integer with the number of elements of the max
            pooling window.
        pool_stride: An integer with the stride of the max pooling window.
        temp_kernel_size: An integer with the size of the temporal max pooling
            window to take a max over.
        temp_stride: An integer with the stride of the temporal max pooling
            window.
        train_out_channels: An integer with the number of channels of the
            training output.
        dropout_p: A float with the probability of an element in the dropout
            layer to be zero-ed.
        noise_std: A float with the standard deviation of the noise layer.
        hidden_nonlinearity: A string with the non-linearity to use at the
            intermediate hidden layers.
        rec_nonlinearity: A string with the non-linearity to use at the locally
            recurrent layer.
        train_out_nonlinearity: A string with the non-linearity to use at the
            training output layer.
        kernel_shape: A string with the shape of the recurrent and max pooling
            kernels.
        temp_pos: A string with the position of the temporal max pooling layer
            with respect to the deep NILRNN.
        temp_kernel_centered: A boolean indicating whether the temporal max
            pooling kernel is centered or applied only over current and past
            samples.
        return_contr: A boolean indicating whether the forward function will
            return the output for the contrastive loss in training mode.
        sparse_vbles: A list of ND tensors of floats with the variables over
            which to apply the sparsity loss.
        sparse_nonlinearity: A string with the non-linearity to use at the
            sparse (locally recurrent) layer.
        dropout_1d: A PyTorch module with the dropout layer.
        noise: A PyTorch module with the noise layer.
        encoder1: A PyTorch module with the hidden layer of the encoder.
        locally_rec_2d: A PyTorch module with the locally recurrent layer.
        max_pool_2d: A PyTorch module with the max pooling layer.
        decoder1: A PyTorch module with the hidden layer of the decoder of
            training mode.
        decoder2: A PyTorch module with the output layer of training mode.
        temp_max_pool: A PyTorch module with the temporal max pooling layer.
        pool_kernel_side: If the kernel is square, an integer with the side of
            the max pooling kernel.
        get_max_pool_input: A function that prepares the input for the max
            pooling layer.
        _temp_pos: A boolean with the position of the temporal max pooling
            layer with respect to the deep NILRNN.
        _train_out_act_fn: An activation function used at the training output.
        _last_mask: A torch tensor with the mask with the output elements to
             be considered in the loss function.
    '''
    def __init__(self, in_features: int, hidden_features: int, rec_shape:
        Tuple[int], rec_kernel_numel: int, pool_kernel_numel: int, pool_stride:
        int, temp_kernel_size: int, temp_stride: int,  train_out_channels: int,
        dropout_p: float = 0, noise_std: float = 0, hidden_nonlinearity: str =
        'relu', rec_nonlinearity: str = 'relu', train_out_nonlinearity: str =
        'relu', kernel_shape: str ='circular', temp_pos: str = 'first',
        temp_kernel_centered: bool = False, return_contr: bool = True) -> None:
        '''Inits DeepLRB.

        Arguments:
            in_features: The size of each input sample.
            hidden_features: The size of the intermediate hidden layers.
            rec_shape: The shape of the locally recurrent layer.
            rec_kernel_numel: The desired number of elements of the recurrent
                connections window.
            pool_kernel_numel: The desired number of elements of the max
                pooling window.
            pool_stride: The stride of the max pooling window.
            temp_kernel_size: The size of the temporal max pooling window to
                take a max over.
            temp_stride: The stride of the temporal max pooling window.
            train_out_channels: The number of channels of the training output.
            dropout_p: The probability of an element in the dropout layer to be
                zero-ed.
            noise_std: The standard deviation of the noise layer.
            hidden_nonlinearity: The non-linearity to use at the intermediate
                hidden layers.
            rec_nonlinearity: The non-linearity to use at the locally recurrent
                layer.
            train_out_nonlinearity: The non-linearity to use at the training
                output layer.
            kernel_shape: The shape of the recurrent and max pooling kernels.
            temp_pos: The position of the temporal max pooling layer with
                respect to the deep NILRNN.
            temp_kernel_centered: The indicator of whether the temporal max
                pooling kernel is centered or applied only over current and
                past samples.
            return_contr: If true, the forward function will return the output
                for the contrastive loss in training mode.
        '''
        super(DeepLRB, self).__init__()
        self.in_features, self.hidden_features, self.rec_shape, \
            self.train_out_channels = in_features, hidden_features, \
            rec_shape, train_out_channels
        self.pool_stride = pool_stride
        self.temp_kernel_size, self.temp_stride = temp_kernel_size, temp_stride
        self.dropout_p, self.noise_std = dropout_p, noise_std
        self.hidden_nonlinearity, self.rec_nonlinearity, \
            self.train_out_nonlinearity = hidden_nonlinearity, \
            rec_nonlinearity, train_out_nonlinearity
        self.kernel_shape = kernel_shape
        self.temp_pos, self.temp_kernel_centered = temp_pos, \
            temp_kernel_centered
        self.return_contr = return_contr
        self.train_out_features = in_features * train_out_channels
        self._last_mask = torch.tensor([])
        self.dropout_1d = nn.Dropout1d(dropout_p)
        self.noise = Noise(noise_std)
        self.encoder1 = nn.Linear(in_features, hidden_features)
        self.sparse_vbles = []
        self.sparse_nonlinearity = rec_nonlinearity
        self.locally_rec_2d = LocallyRec2d(hidden_features, rec_shape,
            rec_kernel_numel, rec_nonlinearity, kernel_shape)
        self.rec_kernel_numel = self.locally_rec_2d.kernel_numel
        if kernel_shape == 'circular':
            self.get_max_pool_input = lambda a: a
            self.max_pool_2d = CircMaxPool2d(rec_shape, pool_kernel_numel,
                pool_stride)
            self.out_features = self.max_pool_2d.out_features
            self.out_shape = self.max_pool_2d.out_shape
            self.pool_kernel_numel = self.max_pool_2d.kernel_numel
        elif kernel_shape == 'square':
            self.pool_kernel_side = round((math.sqrt(pool_kernel_numel) - 1) /
                2) * 2 + 1
            self.pool_kernel_numel = self.pool_kernel_side ** 2
            self.get_max_pool_input = lambda a: a.reshape(-1, *rec_shape)
            self.max_pool_2d = nn.MaxPool2d(self.pool_kernel_side, pool_stride,
                round((self.pool_kernel_side - 1) / 2))
            self.out_shape = (math.ceil(rec_shape[1] / pool_stride), math.ceil(
                rec_shape[1] / pool_stride))
            self.out_features = self.out_shape[0] * self.out_shape[1]
        else:
            raise ValueError(f'{kernel_shape} is not a valid kernel shape')
        self.decoder1 = nn.Linear(math.prod(rec_shape), hidden_features)
        self.decoder2 = nn.Linear(hidden_features, train_out_channels *
            in_features)
        if hidden_nonlinearity == 'sigm':
            self._hidden_act_fn = torch.sigmoid
        elif hidden_nonlinearity == 'tanh':
            self._hidden_act_fn = torch.tanh
        elif hidden_nonlinearity == 'relu':
            self._hidden_act_fn = torch.relu
        else:
            raise ValueError(f'{hidden_nonlinearity} is not a valid '
                             f'non-linearity')
        if train_out_nonlinearity == 'sigm':
            self._train_out_act_fn = torch.sigmoid
        elif train_out_nonlinearity == 'tanh':
            self._train_out_act_fn = torch.tanh
        elif train_out_nonlinearity == 'relu':
            self._train_out_act_fn = torch.relu
        elif train_out_nonlinearity == 'none':
            self._train_out_act_fn = lambda a: a
        else:
            raise ValueError(f'{train_out_nonlinearity} is not a valid '
                             f'non-linearity')
        self.temp_max_pool = TempMaxPool(temp_kernel_size, temp_stride,
            temp_kernel_centered)
        self.scale_factor = self.temp_max_pool.scale_factor
        if temp_pos == 'first':
            self._temp_pos = False
            self.train_scale_factor = self.scale_factor
        elif temp_pos == 'last':
            self._temp_pos = True
            self.train_scale_factor = 1
        else:
            raise ValueError(f'{temp_pos} is not a valid temporal max pooling '
                             f'layer position')
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Initializes weight following Xavier initialization and bias to 0.'''
        nn.init.xavier_uniform_(self.encoder1.weight)
        nn.init.zeros_(self.encoder1.bias)
        self.locally_rec_2d.reset_parameters()
        nn.init.xavier_uniform_(self.decoder1.weight)
        nn.init.zeros_(self.decoder1.bias)
        nn.init.xavier_uniform_(self.decoder2.weight)
        nn.init.zeros_(self.decoder2.bias)

    def _deep_nilrnn_forward(self, x: torch.Tensor, h_0: torch.Tensor = None) \
        -> Tuple[torch.Tensor]:
        '''Performs a forward pass without the temporal max pooling layer.

        Arguments:
            x: The input batch.
            h_0: The initial hidden state. Defaults to zeros if not provided.

        Returns:
            pred: The output batch (or prediction).
            pred2: An additional output batch for contrastive loss if required.
            y: The desired output batch (only in training mode).
            h_n: The final hidden state.
        '''
        del self.sparse_vbles[:]
        a = self.locally_rec_2d(self._hidden_act_fn(self.encoder1(self.noise(
            self.dropout_1d(x.T).T))), h_0)
        self.sparse_vbles.append(a)
        if self.training:
            pred = self._train_out_act_fn(self.decoder2(self._hidden_act_fn(
                self.decoder1(a))))
            y = F.pad(x.T, (0, self.train_out_channels - 1), mode=
                'replicate').T.contiguous().as_strided((x.shape[0],
                self.train_out_channels * self.in_features), (self.in_features,
                    1))
            if self.return_contr:
                return pred, self.max_pool_2d(self.get_max_pool_input(
                    a)).reshape(-1, self.out_features), y, a[-1]
            else:
                return pred, y, a[-1]
        else:
            return self.max_pool_2d(self.get_max_pool_input(a)).reshape(-1,
                self.out_features), a[-1]

    def forward(self, x: torch.Tensor, rec_0: torch.Tensor = None, temp_0:
        torch.Tensor = None) -> Tuple[torch.Tensor]:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.
            rec_0: The initial locally recurrent layer hidden state. Defaults
                to zeros if not provided.
            temp_0: The initial temporal max pooling layer hidden state.
                Defaults to -infs if not provided.

        Returns:
            pred: The output batch (or prediction).
            pred2: An additional output batch for contrastive loss if required.
            y: The desired output batch (only in training mode).
            rec_n: The final locally recurrent layer hidden state.
            temp_n: The final temporal max pooling layer hidden state.
        '''
        if self._temp_pos:
            *output, rec_n = self._deep_nilrnn_forward(x, rec_0)
            if self.training:
                return tuple(output) + (rec_n, None)
            else:
                pred, temp_n = self.temp_max_pool(output[0], temp_0)
                return pred, rec_n, temp_n
        else:
            a, temp_n = self.temp_max_pool(x, temp_0)
            *output, rec_n = self._deep_nilrnn_forward(a, rec_0)
            return tuple(output) + (rec_n, temp_n)

    def get_mask(self, batch_size: int) -> torch.Tensor:
        '''Returns a mask for the loss function.

        Arguments:
            batch_size: The size of the input batch (after the temporal max
                pooling layer).

        Returns:
            mask: The mask with the output elements to be considered in the
                loss function.
        '''
        if batch_size == self._last_mask.shape[0]:
            return self._last_mask
        mask = torch.ones(batch_size, self.train_out_channels *
            self.in_features)
        for i in range(1, min(self.train_out_channels, batch_size + 1)):
            mask[-i, i * self.in_features:] = 0
        self._last_mask = mask
        return mask


class MDN(nn.Module):
    '''A shallow feedforward Mixture Density Network of Gaussian distributions.

    Attributes:
        in_features: An integer with the size of each input sample.
        hidden_features: An integer with the size of the hidden layer.
        out_features: An integer with the size of each output sample.
        n_components: An integer with the number of Gaussian distributions
            sumed.
        hidden: A PyTorch module with the hidden layer.
        mean: A PyTorch module with the output mean layer.
        var: A PyTorch module with the output variance layer.
        coef: A PyTorch module with the output layer with the component mixing
            coefficients (or probabilities).
    '''
    def __init__(self, in_features: int, hidden_features: int, out_features:
        int, n_components: int) -> None:
        '''Inits MDN.

        Arguments:
            in_features: The size of each input sample.
            hidden_features: The size of the hidden layer.
            out_features: The size of each output sample.
            n_components: The number of rectified Gaussian distributions sumed.
        '''
        super(MDN, self).__init__()
        self.in_features, self.hidden_features = in_features, hidden_features
        self.out_features, self.n_components = out_features, n_components
        self.hidden = nn.Linear(in_features, hidden_features)
        self.mean = nn.Linear(hidden_features, n_components * out_features)
        self.var = nn.Linear(hidden_features, n_components * out_features)
        self.coef = nn.Linear(hidden_features, n_components)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Initializes weight following Xavier initialization and bias.'''
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.mean.weight)
        self.mean.bias = nn.Parameter(torch.randn_like(self.mean.bias))
        nn.init.xavier_uniform_(self.var.weight)
        nn.init.zeros_(self.var.bias)
        nn.init.xavier_uniform_(self.coef.weight)
        nn.init.zeros_(self.coef.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.

        Returns:
            mn: The output mean batch (or prediction).
            vr: The output variance batch (or prediction).
            p: The output component mixing coefficient batch (or prediction).
        '''
        a = self.hidden(x).relu()
        mn = self.mean(a).clamp(-100, 100).reshape(-1, self.n_components,
            self.out_features)
        vr = (F.elu(self.var(a)) + 1 + 1e-4).reshape(-1, self.n_components,
            self.out_features)
        p = self.coef(a).softmax(-1) + 1e-5
        return mn, vr, p


class LRBPredictor(nn.Module):
    '''A shallow feedforward Gaussian MDN to predict LRB samples.

    Attributes:
        main_features: An integer with the number of dimensions of each
            Gaussian distribution in the output mixture.
        main_n_components: An integer with the number of sumed Gaussian
            distributions in the output mixture.
        hidden_features: An integer with the size of the hidden layer.
        ctxt_features: An integer with the number of dimensions of each
            Gaussian distribution in the input mixture.
        ctxt_n_components: An integer with the number of sumed Gaussian
            distributions in the input mixture (None when the future context
            information is a sample).
        ctxt_horizon: An integer with the number of timesteps ahead of the
            context prediction (ignored when the future context information
            is a distribution).
        horizon: An integer with the number of timesteps ahead of the
            prediction.
        rho: A float with the expected sparsity parameter (used at
            initialization).
        use_actual_ctxt: A boolean indicating whether the actual future context
            samples should be used as input instead of the predicted/selected
            ones.
        mdn: A PyTorch module with the MDN.
        _last_mask: A torch tensor with the mask with the output elements to
             be considered in the loss function.
    '''
    def __init__(self, main_features: int, main_n_components: int,
        hidden_features: int, ctxt_features: int, ctxt_n_components: int =
        None, ctxt_horizon: int = None, horizon: int = 1, rho: float = .02) \
        -> None:
        '''Inits LRBPredictor.

        Arguments:
            main_features: The number of dimensions of each Gaussian
                distribution in the output mixture.
            main_n_components: The number of sumed Gaussian distributions in
                the output mixture.
            hidden_features: The size of the hidden layer.
            ctxt_features: The number of dimensions of each Gaussian
                distribution in the input mixture.
            ctxt_n_components: The number of sumed Gaussian distributions in
                the input mixture (None when the future context information is
                a sample).
            ctxt_horizon: The number of timesteps ahead of the context
                prediction (ignored when the future context information is a
                distribution).
            horizon: The number of timesteps ahead of the prediction.
            rho: The expected sparsity parameter (used at initialization).
        '''
        super(LRBPredictor, self).__init__()
        self.main_features, self.main_n_components = main_features, \
            main_n_components
        self.hidden_features = hidden_features
        self.ctxt_features, self.ctxt_n_components = ctxt_features, \
            ctxt_n_components
        self.ctxt_horizon, self.horizon = ctxt_horizon, horizon
        self.rho = rho
        self._last_mask = torch.tensor([])
        if ctxt_n_components is None:
            mdn_in_features = main_features + 2 * ctxt_features
            self.use_actual_ctxt = True
        else:
            mdn_in_features = main_features + ctxt_features + 2 * \
                ctxt_n_components * ctxt_features + ctxt_n_components
            self.use_actual_ctxt = False
        self.mdn = MDN(mdn_in_features, hidden_features, main_features,
            main_n_components)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Initializes weight following Xavier initialization and bias.'''
        self.mdn.reset_parameters()
        self.mdn.mean.bias = nn.Parameter((torch.rand_like(self.mdn.mean.bias)
            < self.rho).float())

    def forward(self, x_main: torch.Tensor, *x_ctxt: torch.Tensor) -> Tuple[
        torch.Tensor]:
        '''Performs a forward pass.

        Arguments:
            x_main: The part to be predicted of the input batch.
            ctxt_cur: The part with the current context samples of the input
                batch.
            ctxt_mn: The part with the predicted context mixture means of the
                input batch (or the selected future context samples if
                ctxt_n_components is None).
            ctxt_vr: The part with the predicted context mixture variances of
                the input batch (if ctxt_n_components is not None).
            ctxt_p: The part with the predicted context mixture coefficients of
                the input batch (if ctxt_n_components is not None).

        Returns:
            mn: The output mean batch (or prediction).
            vr: The output variance batch (or prediction).
            p: The output component mixing coefficient batch (or prediction).
            y: The desired output batch (only in training mode).
        '''
        if self.use_actual_ctxt:
            x_ctxt = (x_ctxt[0], F.pad(x_ctxt[0].T, (0, self.ctxt_horizon),
                mode='replicate').T[self.ctxt_horizon:])
        mn, vr, p = self.mdn(torch.cat([x_main] + [c.reshape(x_main.shape[0],
            -1) for c in x_ctxt], -1))
        if self.training:
            y = F.pad(x_main.T, (0, self.horizon), mode='replicate').T[
                self.horizon:]
            return mn, vr, p, y
        else:
            return mn, vr, p

    def get_mask(self, batch_size: int) -> torch.Tensor:
        '''Returns a mask for the loss function.

        Arguments:
            batch_size: The size of the input batch (after the temporal max
                pooling layer).

        Returns:
            mask: The mask with the output elements to be considered in the
                loss function.
        '''
        if batch_size == self._last_mask.shape[0]:
            return self._last_mask
        mask = torch.ones(batch_size)
        mask[-self.horizon:] = 0
        self._last_mask = mask
        return mask


class GRULRBPredictor(nn.Module):
    '''A shallow GRU + feedforward Gaussian MDN to predict LRB samples.

    Attributes:
        main_features: An integer with the number of dimensions of each
            Gaussian distribution in the output mixture.
        main_n_components: An integer with the number of sumed Gaussian
            distributions in the output mixture.
        gru_features: An integer with the size of the GRU hidden layer.
        hidden_features: An integer with the size of the feedforward hidden
            layer.
        ctxt_features: An integer with the size of the part with context
            information of each input sample.
        horizon: An integer with the number of timesteps ahead of the
            prediction.
        rho: A float with the expected sparsity parameter (used at
            initialization).
        gru: A PyTorch module with the GRU layer.
        mdn: A PyTorch module with the MDN.
        _last_mask: A torch tensor with the mask with the output elements to
             be considered in the loss function.
    '''
    def __init__(self, main_features: int, main_n_components: int,
        gru_features: int, hidden_features: int, ctxt_features: int = None,
        horizon: int = 1, rho: float = .02) -> None:
        '''Inits GRULRBPredictor.

        Arguments:
            main_features: The number of dimensions of each Gaussian
                distribution in the output mixture.
            main_n_components: The number of sumed Gaussian distributions in
                the output mixture.
            gru_features: The size of the GRU hidden layer.
            hidden_features: The size of the feed forward hidden layer.
            ctxt_features: The size of the part with context information
                of each input sample.
            horizon: The number of timesteps ahead of the prediction.
            rho: The expected sparsity parameter (used at initialization).
        '''
        super(GRULRBPredictor, self).__init__()
        self.main_features, self.main_n_components = main_features, \
            main_n_components
        self.gru_features, self.hidden_features = gru_features, hidden_features
        self.ctxt_features = ctxt_features
        self.horizon = horizon
        self.rho = rho
        self._last_mask = torch.tensor([])
        if ctxt_features is None:
            ctxt_features = 0
        self.gru = nn.GRU(main_features + ctxt_features, gru_features)
        self.mdn = MDN(main_features + ctxt_features + gru_features,
            hidden_features, main_features, main_n_components)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Initializes weight following Xavier initialization and bias.'''
        self.gru.reset_parameters()
        self.mdn.reset_parameters()
        self.mdn.mean.bias = nn.Parameter((torch.rand_like(self.mdn.mean.bias)
            < self.rho).float())

    def forward(self, *xs: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Performs a forward pass.

        Arguments:
            x_main: The part to be predicted of the input batch.
            x_ctxt: The part with the context information of the input batch
                (if ctxt_features is not None).
            h_0: The initial hidden state. Defaults to zeros if not provided.

        Returns:
            mn: The output mean batch (or prediction).
            vr: The output variance batch (or prediction).
            p: The output component mixing coefficient batch (or prediction).
            y: The desired output batch (only in training mode).
            h_n: The final hidden state.
        '''
        #if not self.gru.weight_ih_l0.isfinite().all() and \
        #    self.mdn.hidden.weight.isfinite().all():
        #    raise ValueError('GRU weights diverged due to exploding gradients.'
        #                     ' Consider lowering the learning rate')
        x_main = xs[0]
        n_inputs = 1 if self.ctxt_features is None else 2
        x = torch.cat(xs[:n_inputs], -1)
        h_0 = torch.zeros(1, self.gru_features) if len(xs) == n_inputs else \
            xs[-1]
        a, h_n = self.gru(x, h_0)
        mn, vr, p = self.mdn(torch.cat([x, a], -1))
        if self.training:
            y = F.pad(x_main.T, (0, self.horizon), mode='replicate').T[
                self.horizon:]
            return mn, vr, p, y, h_n
        else:
            return mn, vr, p, h_n

    def get_mask(self, batch_size: int) -> torch.Tensor:
        '''Returns a mask for the loss function.

        Arguments:
            batch_size: The size of the input batch (after the temporal max
                pooling layer).

        Returns:
            mask: The mask with the output elements to be considered in the
                loss function.
        '''
        if batch_size == self._last_mask.shape[0]:
            return self._last_mask
        mask = torch.ones(batch_size)
        mask[-self.horizon:] = 0
        self._last_mask = mask
        return mask


class UpsampleLinear(nn.Module):
    '''An upsampling linear neural network.

    Attributes:
        scale_factor: An integer with the multiplier for the batch size.
        in_features: An integer with the size of each input sample.
        out_features: An integer with the size of each output sample.
        upsample_pos: A string with the position of the upsample layer with
            respect to the linear layer.
        upsample_mode: A string with the algorithm used for upsampling.
        train_scale_factor: An integer with the multiplier for the batch size
            in training mode.
        upsample: A PyTorch module with the upsample layer.
        linear: A PyTorch module with the linear layer.
        _upsample_pos: A boolean with the position of the upsample layer with
            respect to the linear layer.
    '''
    def __init__(self, scale_factor: int, in_features: int, out_features: int,
        upsample_pos: str = 'first', upsample_mode: str = 'constant') -> None:
        '''Inits UpsampleLinear.

        Arguments:
            scale_factor: The multiplier for the batch size.
            in_features: The size of each input sample.
            out_features: The size of each output sample.
            upsample_pos: The position of the upsample layer with respect to
                the linear layer.
            upsample_mode: The algorithm used for upsampling.
        '''
        super(UpsampleLinear, self).__init__()
        self.scale_factor = scale_factor
        self.in_features, self.out_features = in_features, out_features
        self.upsample_pos, self.upsample_mode = upsample_pos, upsample_mode
        self.upsample = Upsample(scale_factor, upsample_mode)
        self.linear = nn.Linear(in_features, out_features)
        if upsample_pos == 'first':
            self._upsample_pos = False
            self.train_scale_factor = scale_factor
        elif upsample_pos == 'last':
            self._upsample_pos = True
            self.train_scale_factor = 1
        else:
            raise ValueError(f'{upsample_pos} is not a valid upsample layer '
                             f'position')
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Initializes weight following Xavier initialization and bias to 0.'''
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.

        Returns:
            pred: The output batch (or prediction).
        '''
        if self._upsample_pos:
            return self.upsample(self.linear(x))
        else:
            return self.linear(self.upsample(x))


class ShallowRNN(nn.Module):
    '''A shallow vanilla recurrent neural network.

    Attributes:
        in_features: An integer with the size of each input sample.
        hidden_features: An integer with the size of the recurrent layer.
        out_features: An integer with the size of each output sample.
        nonlinearity: A string with the non-linearity to use.
        rec: A PyTorch module with the hidden recurrent layer.
        get_output: A function that extracts the output and hidden state of the
            recurrent layer.
        linear: A Pytorch module with the output linear layer.
    '''
    def __init__(self, in_features: int, hidden_features: int, out_features:
        int, nonlinearity: str = 'tanh') -> None:
        '''Inits ShallowRNN.

        Arguments:
            in_features: The size of each input sample.
            hidden_features: The size of the recurrent layer.
            out_features: The size of each output sample.
            nonlinearity: The non-linearity to use.
        '''
        super(ShallowRNN, self).__init__()
        self.in_features, self.hidden_features, self.out_features, \
            self.nonlinearity = in_features, hidden_features, out_features, \
            nonlinearity
        if nonlinearity == 'sigm':
            self.rec = SigmRec(in_features, hidden_features)
            self.get_output = lambda a: (a, a[-1])
        else:
            self.rec = nn.RNN(in_features, hidden_features,
                nonlinearity=nonlinearity)
            self.get_output = lambda a: a
        self.linear = nn.Linear(hidden_features, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Initializes weight following Xavier initialization and bias to 0.'''
        if self.nonlinearity == 'sigm':
            self.rec.reset_parameters()
        else:
            weight = nn.init.xavier_uniform_(torch.empty(self.hidden_features,
                self.in_features + self.hidden_features))
            self.rec.weight_ih_l0 = nn.Parameter(weight[:,:self.in_features])
            self.rec.weight_hh_l0 = nn.Parameter(weight[:,self.in_features:])
            nn.init.zeros_(self.rec.bias_ih_l0)
            nn.init.zeros_(self.rec.bias_hh_l0)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None) -> \
        torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.
            h_0: The initial hidden state. Defaults to zeros if not provided.

        Returns:
            pred: The output batch (or prediction).
            h_n: The final hidden state.
        '''
        a, h_n = self.get_output(self.rec(x, h_0))
        return self.linear(a), h_n


class UpsampleRNN(nn.Module):
    '''An upsampling shallow vanilla recurrent neural network.

    Attributes:
        scale_factor: An integer with the multiplier for the batch size.
        in_features: An integer with the size of each input sample.
        hidden_features: An integer with the size of the recurrent layer.
        out_features: An integer with the size of each output sample.
        nonlinearity: A string with the non-linearity to use.
        upsample_pos: A string with the position of the upsample layer with
            respect to the RNN.
        upsample_mode: A string with the algorithm used for upsampling.
        train_scale_factor: An integer with the multiplier for the batch size
            in training mode.
        upsample: A PyTorch module with the upsample layer.
        rnn: A PyTorch module with the shallow RNN.
        _upsample_pos: A boolean with the position of the upsample layer with
            respect to the RNN.
    '''
    def __init__(self, scale_factor: int, in_features: int, hidden_features:
        int, out_features: int, nonlinearity: str = 'tanh', upsample_pos: str
        = 'first', upsample_mode: str = 'constant') -> None:
        '''Inits UpsampleRNN.

        Arguments:
            scale_factor: The multiplier for the batch size.
            in_features: The size of each input sample.
            hidden_features: The size of the recurrent layer.
            out_features: The size of each output sample.
            nonlinearity: The non-linearity to use.
            upsample_pos: The position of the upsample layer with respect to
                the RNN.
            upsample_mode: The algorithm used for upsampling.
        '''
        super(UpsampleRNN, self).__init__()
        self.scale_factor = scale_factor
        self.in_features, self.hidden_features, self.out_features, \
            self.nonlinearity = in_features, hidden_features, out_features, \
            nonlinearity
        self.upsample_pos, self.upsample_mode = upsample_pos, upsample_mode
        self.upsample = Upsample(scale_factor, upsample_mode)
        self.rnn = ShallowRNN(in_features, hidden_features, out_features,
            nonlinearity)
        if upsample_pos == 'first':
            self._upsample_pos = False
            self.train_scale_factor = scale_factor
        elif upsample_pos == 'last':
            self._upsample_pos = True
            self.train_scale_factor = 1
        else:
            raise ValueError(f'{upsample_pos} is not a valid upsample layer '
                             f'position')

    def reset_parameters(self) -> None:
        '''Initializes weight following Xavier initialization and bias to 0.'''
        self.rnn.reset_parameters()

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None) -> \
        torch.Tensor:
        '''Performs a forward pass.

        Arguments:
            x: The input batch.
            h_0: The initial hidden state. Defaults to zeros if not provided.

        Returns:
            pred: The output batch (or prediction).
            h_n: The final hidden state.
        '''
        if self._upsample_pos:
            a, h_n = self.rnn(x, h_0)
            return self.upsample(a), h_n
        else:
            return self.rnn(self.upsample(x), h_0)


class HLRNN(nn.Module):
    '''The Hierarchical Locally Recurrent Neural Network (stacks LRBs).

    Attributes:
        blocks: A dictionary of strings to PyTorch modules with the neural
            network block names and blocks forming the hierarchy.
        input_names: A list of strings with the names of the inputs to the
            neural network.
        block_inputs: A dictionary of strings to strings or lists of strings
            with the neural network block names and input names.
        hidden_names: A list of strings with the names of the hidden states of
            the neural network.
        block_hiddens: A dictionary of strings to strings or lists of strings
            with the neural network block names and hidden state names.
        cur_block_name: A string with the current output block name.
        cur_block: A PyTorch module with the current output block.
        closed_loop_block_name: A string with the current closed-loop output
            block name.
        upsample_out: A boolean with the indicator of whether the output should
            be upsampled.
        upsample_mode: A string with the algorithm used for upsampling.
        upsample: A PyTorch module with either the Upsample layer or the
            Identity layer.
        scale_factors: A dictionary of strings to floats with the neural
            network block names and multipliers for the batch size.
        train_scale_factors: A dictionary of strings to floats with the neural
            network block names and multipliers for the batch size at the
            training layer(s).
        scale_factor: A float with the current multiplier for the batch size.
        train_scale_factor: A float with the current multiplier for the batch
            size at the training layer(s).
        cur_min_scale_factor: A float with the minimum multiplier for the batch
            size among the blocks currently being used.
        get_mask: A function that returns the current mask for the loss
            function.
        _blocks: A list of PyTorch modules with the neural network blocks
            forming the hierarchy.
        _block_input_ids: A list of lists of ints with the neural network block
            input identifiers.
        _block_hidden_ids: A list of lists of ints with the neural network
            block output hidden state identifiers.
        _closed_loop_buffer_lens: A list of integers with the length at which
            the input buffers should be inputted to the upsampling blocks
            during closed-loop operations.
        _closed_loop_strides: A list of integers with the time between
            considered samples at each block during closed-loop operations.
        _cur_block_id: An integer with the current output block identifier.
        _closed_loop_block_id: An integer with the current closed-loop output
            block identifier.
        _blocks_are_cur: A list of booleans indicating the neural network
            blocks currently being used.
    '''
    def __init__(self, blocks: typing.OrderedDict[str, nn.Module], input_names:
        List[str], block_inputs: Dict[str, List[str]], hidden_names: List[str]
        = [], block_hiddens: Dict[str, List[str]] = {}) -> None:
        '''Inits HLRNN.

        Arguments:
            blocks: The neural network block names and blocks forming the
                hierarchy.
            input_names: The names of the inputs to the neural network.
            block_inputs: The neural network block names and input names.
            hidden_names: The names of the hidden states of the neural network.
            block_hiddens: The neural network block names and hidden state
                names.
        '''
        super(HLRNN, self).__init__()
        self.blocks = OrderedDict()
        self.input_names = input_names
        self.block_inputs = {}
        self.hidden_names = []
        self.block_hiddens = {}
        self._blocks = []
        self._block_input_ids = []
        self._block_hidden_ids = []
        self.upsample_out = False
        self.upsample_mode = 'constant'
        self.scale_factors = {name: 1 for name in input_names}
        self.train_scale_factors = {}
        self._closed_loop_buffer_lens = []
        self._closed_loop_strides = []
        self.cur_block = None
        self.unset_cur_block()
        self.unset_closed_loop_block()
        self.extend(blocks, block_inputs, hidden_names, block_hiddens)

    def forward(self, *xs: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Performs a forward pass.

        Arguments:
            xs: The input batches and initial hidden states.

        Returns:
            preds: The output batches (or predictions) and final hidden states.
        '''
        outputs = [(x,) for x in xs] + [()] * (len(self.input_names) + len(
            self.hidden_names) - len(xs)) + [None] * len(self._blocks)
        hiddens = [None] * len(self.hidden_names)
        for output_id, (block, input_id, hidden_id, is_cur) in enumerate(zip(
            self._blocks, self._block_input_ids, self._block_hidden_ids,
            self._blocks_are_cur), len(self.input_names) + len(
            self.hidden_names)):
            if is_cur:
                output = block(*(out for in_id in input_id for out in outputs[
                    in_id]))
                if not isinstance(output, tuple):
                    output = (output,)
                elif len(hidden_id) != 0:
                    for hid_id, hid in zip(hidden_id, output[-len(
                        hidden_id):]):
                        hiddens[hid_id] = hid
                    output = output[:-len(hidden_id)]
                outputs[output_id] = output
        output = outputs[self._cur_block_id + len(self.input_names) + len(
            self.hidden_names)]
        if len(output) == 1 and len(hiddens) == 0:
            return self.upsample(output[0])
        else:
            return tuple(self.upsample(out) for out in output) + tuple(hiddens)

    def closed_loop_forward(self, *xs: torch.Tensor, horizon: int,
        repetitions: int) -> Tuple[torch.Tensor]:
        '''Performs a closed-loop forward pass.

        Arguments:
            xs: The input batches and initial hidden states.
            horizon: The number of steps forward to perform starting from each
                time sample.
            repetitions: The number of times the forward pass is repeated.

        Returns:
            preds: The output batches (or predictions) and final hidden states.
        '''
        hs = [(h,) for h in xs[len(self.input_names):]]
        if len(hs) > len(self.hidden_names):
            hs = hs[:len(self.hidden_names)]
            in_buffers = xs[-3]
            out_buffers = xs[-2]
            next_times = xs[-1]
            in_buffer_lens = self._closed_loop_buffer_lens
        else:
            hs = hs + [(None,)] * (len(self.hidden_names) - len(hs))
            in_buffers = [[] if hasattr(block, 'scale_factor') and
                block.scale_factor < 1 else None for block in self._blocks]
            out_buffers = [(torch.tensor([]),) if hasattr(block,
                'scale_factor') and block.scale_factor > 1 else None for block
                in self._blocks]
            next_times = [0] * len(self._blocks)
            in_buffer_lens = [1] * len(self._blocks)
        xs = [x.unsqueeze(1) for x in xs[:len(self.input_names)]]
        pred_list = []
        for samples in zip(*xs):
            samples = [(x,) for x in samples] + hs
            iter_list = []
            for _ in range(repetitions):
                iter_list.append(self._closed_loop_iteration(samples, horizon,
                    in_buffers, out_buffers, next_times, in_buffer_lens))
            pred_list.append(tuple(torch.stack(iter) for iter in zip(
                *iter_list)))
            pred_xs, _ = self._closed_loop_step(samples, in_buffers,
                out_buffers, next_times, in_buffer_lens)
            hs = pred_xs[len(self.input_names):]
            in_buffer_lens = self._closed_loop_buffer_lens
        output = tuple(torch.stack(pred) for pred in zip(*pred_list))
        return output + tuple(h[0] for h in hs) + (in_buffers, out_buffers,
            next_times)

    def closed_loop_forward_step(self, *xs: torch.Tensor) -> Tuple[
        torch.Tensor]:
        '''Performs a closed-loop forward pass.

        Arguments:
            xs: The input batches and current-step hidden states.

        Returns:
            preds: The output batches (or predictions) and next-step hidden
                states.
        '''
        if len(xs) > len(self.input_names) + len(self.hidden_names):
            in_buffers = xs[-3]
            out_buffers = xs[-2]
            next_times = xs[-1]
            xs = [(x,) for x in xs[:len(self.input_names) + len(
                self.hidden_names)]]
            in_buffer_lens = self._closed_loop_buffer_lens
        else:
            xs = [(x,) for x in xs] + [(None,)] * (len(self.input_names) + len(
                self.hidden_names) - len(xs))
            in_buffers = [[] if hasattr(block, 'scale_factor') and
                block.scale_factor < 1 else None for block in self._blocks]
            out_buffers = [(torch.tensor([]),) if hasattr(block,
                'scale_factor') and block.scale_factor > 1 else None for block
                in self._blocks]
            next_times = [0] * len(self._blocks)
            in_buffer_lens = [1] * len(self._blocks)
        xs, _ = self._closed_loop_step(xs, in_buffers, out_buffers, next_times,
            in_buffer_lens)
        return tuple(out[0] for out in xs) + (in_buffers, out_buffers,
            next_times)

    def _closed_loop_iteration(self, xs: List[torch.Tensor], horizon: int,
        in_buffers: List[List[Tuple[torch.Tensor]]], out_buffers: List[Tuple[
        torch.Tensor]], next_times: List[int], in_buffer_lens_0: List[int]) \
        -> Tuple[torch.Tensor]:
        '''Performs a closed-loop forward pass starting from the given sample.

        Arguments:
            xs: The input single-sample batches and initial hidden states.
            horizon: The number of steps forward to perform.
            in_buffers: The stored samples at the input of the downsampling
                blocks.
            out_buffers: The stored samples at the output of the upsampling
                blocks.
            next_times: The number of samples left for each block to be used.
            in_buffer_lens_0: The length at which the input buffers should be
                inputted to the upsampling blocks in the first step.

        Returns:
            preds: The output batches (or predictions).
        '''
        in_buffers, out_buffers, next_times = copy.deepcopy(in_buffers), \
            copy.copy(out_buffers), copy.copy(next_times)
        in_buffer_lens = in_buffer_lens_0
        pred_list = []
        for step in range(horizon):
            xs, preds = self._closed_loop_step(xs, in_buffers, out_buffers,
                next_times, in_buffer_lens)
            pred_list.append(preds)
            in_buffer_lens = self._closed_loop_buffer_lens
        return tuple(self.upsample(torch.cat(pred, 0)) for pred in zip(
            *pred_list))

    def _closed_loop_step(self, xs: List[torch.Tensor], in_buffers: List[List[
        Tuple[torch.Tensor]]], out_buffers: List[Tuple[torch.Tensor]],
        next_times: List[int], in_buffer_lens: List[int]) -> Tuple[List[
        torch.Tensor], Tuple[torch.Tensor]]:
        '''Performs a one-step forward pass for closed-loop operations.

        Arguments:
            xs: The input single-sample batches and initial hidden states.
            in_buffers: The stored samples at the input of the downsampling
                blocks.
            out_buffers: The stored samples at the output of the upsampling
                blocks.
            next_times: The number of samples left for each block to be used.
            in_buffer_lens: The length at which the input buffers should be
                inputted to the upsampling blocks.

        Returns:
            xs: The input single-sample batches and initial hidden states of
                the next step.
            preds: The output batches (or predictions).
        '''
        outputs = xs + [None] * len(self._blocks)
        hiddens = xs[len(self.input_names):]
        for block_id, (block, input_id, hidden_id, in_buffer, buffer_len,
            stride, is_cur) in enumerate(zip(self._blocks,
            self._block_input_ids, self._block_hidden_ids, in_buffers,
            in_buffer_lens, self._closed_loop_strides, self._blocks_are_cur)):
            if is_cur and next_times[block_id] == 0:
                next_times[block_id] += stride
                output = None
                if in_buffer is not None:
                    input_x_id = input_id[:len(input_id) - len(hidden_id)]
                    in_buffer.append(tuple(out for in_id in input_x_id for out
                        in outputs[in_id]))
                    if len(in_buffer) == buffer_len:
                        input_h_id = input_id[len(input_id) - len(hidden_id):]
                        output = block(*(torch.cat(x, 0) for x in zip(
                            *in_buffer)), *(out for in_id in input_h_id for out
                            in outputs[in_id]), shift=buffer_len - 1)
                        in_buffer.clear()
                elif out_buffers[block_id] is None or len(out_buffers[
                    block_id][0]) == 0:
                    output = block(*(out for in_id in input_id for out in
                        outputs[in_id]))
                if output is not None:
                    if not isinstance(output, tuple):
                        output = (output,)
                    elif len(hidden_id) != 0:
                        for hid_id, hid in zip(hidden_id, output[-len(
                            hidden_id):]):
                            hiddens[hid_id] = (hid,)
                        output = output[:-len(hidden_id)]
                if out_buffers[block_id] is not None:
                    if output is not None:
                        out_buffers[block_id] = output
                    output = tuple(out[:1] for out in out_buffers[block_id])
                    out_buffers[block_id] = tuple(out[1:] for out in
                        out_buffers[block_id])
                outputs[block_id + len(self.input_names) + len(
                    self.hidden_names)] = output
            next_times[block_id] -= 1
        xs = [(out,) for out in outputs[self._closed_loop_block_id + len(
            self.input_names) + len(self.hidden_names)]]
        preds = outputs[self._cur_block_id + len(self.input_names) + len(
            self.hidden_names)]
        return xs + hiddens, preds

    def extend(self, blocks: typing.OrderedDict[str, nn.Module], block_inputs:
        Dict[str, List[str]], hidden_names: List[str] = [], block_hiddens:
        Dict[str, List[str]] = {}) -> None:
        '''Extends the hierarchy with the given blocks.

        Arguments:
            blocks: The new neural network block names and blocks extending the
                hierarchy.
            block_inputs: The new neural network block names and input names.
            hidden_names: The names of the hidden states of the neural network.
            block_hiddens: The neural network block names and hidden state
                names.
        '''
        for input_id in self._block_input_ids:
            for i, in_id in enumerate(input_id):
                if in_id >= len(self.input_names) + len(self.hidden_names):
                    input_id[i] += len(hidden_names)
        self.blocks.update(blocks)
        self.block_inputs.update(block_inputs)
        self.hidden_names.extend(hidden_names)
        self.block_hiddens.update(block_hiddens)
        self._blocks.extend(list(blocks.values()))
        for block in blocks.values():
            for param in block.parameters():
                param.requires_grad = False
        block_input_ids = {name: id for id, name in enumerate(
            self.input_names + self.hidden_names + list(self.blocks))}
        block_hidden_ids = {name: id for id, name in enumerate(
            self.hidden_names)}
        for block_name in blocks:
            block_input = block_inputs[block_name]
            if isinstance(block_input, str):
                block_input = [block_input]
            block_hidden = []
            if block_name in block_hiddens:
                block_hidden = block_hiddens[block_name]
                if isinstance(block_hidden, str):
                    block_hidden = [block_hidden]
            self._block_input_ids.append([block_input_ids[block_in] for
                block_in in block_input + block_hidden])
            self._block_hidden_ids.append([block_hidden_ids[block_hid] for
                block_hid in block_hidden])
        for block_name, block in blocks.items():
            block_input = block_inputs[block_name]
            if isinstance(block_input, str):
                scale_factor = self.scale_factors[block_input]
            else:
                scale_factor = self.scale_factors[block_input[0]]
            self.scale_factors[block_name] = scale_factor
            if hasattr(block, 'scale_factor'):
                self.scale_factors[block_name] *= block.scale_factor
            self.train_scale_factors[block_name] = scale_factor
            if hasattr(block, 'train_scale_factor'):
                self.train_scale_factors[block_name] *= \
                    block.train_scale_factor
        self._closed_loop_buffer_lens.extend([round(1 / block.scale_factor) if
            hasattr(block, 'scale_factor') and block.scale_factor < 1 else None
            for block in blocks.values()])
        self._closed_loop_strides.extend([round(block.scale_factor /
            self.scale_factors[name]) if hasattr(block, 'scale_factor') and
            block.scale_factor < 1 else round(1 / self.scale_factors[name]) for
            name, block in blocks.items()])
        self._blocks_are_cur.extend([False] * len(blocks))
        for block_name, block in blocks.items():
            setattr(self, block_name, block)

    def extend_input(self, input_names: List[str]) -> None:
        '''Extends the inputs to the hierarchy with the given input names.

        Arguments:
            input_names: The names of the new inputs to the neural network.
        '''
        for input_id in self._block_input_ids:
            for i, in_id in enumerate(input_id):
                if in_id >= len(self.input_names):
                    input_id[i] += len(input_names)
        self.input_names.extend(input_names)
        self.scale_factors.update({name: 1 for name in input_names})

    def get_sub_hlrnn(self, block_names: List[str], input_names: List[str]) \
        -> Self:
        '''Returns the smallest HLRNN contained in the HLRNN up to constraints.

        Arguments:
            block_names: The neural network block names that need to be
                included in the new neural network.
            input_names: The input or neural network block names that are input
                to the new neural network (blocks are not included).

        Returns:
            sub_hlrnn: The new neural network.
        '''
        sub_block_names = [name for name in block_names]
        for name in sub_block_names:
            inputs = self.block_inputs[name]
            if isinstance(inputs, str):
                inputs = [inputs]
            for input in inputs:
                if input not in sub_block_names and input not in input_names:
                    sub_block_names.append(input)
        sub_block_names = [name for name in list(self.blocks) if name in
            sub_block_names]
        sub_blocks = OrderedDict([(name, self.blocks[name]) for name in
            sub_block_names])
        sub_block_inputs = {name: self.block_inputs[name] for name in
            sub_block_names}
        sub_block_hiddens = {name: self.block_hiddens[name] for name in
            sub_block_names if name in self.block_hiddens}
        sub_hidden_names = [name for names in [[names] if isinstance(names,
            str) else names for names in sub_block_hiddens.values()] for name
            in names]
        return type(self)(sub_blocks, input_names, sub_block_inputs,
            sub_hidden_names, sub_block_hiddens)

    def set_cur_block(self, block_name: str, upsample_out: bool = None,
        upsample_mode: str = None) -> None:
        '''Sets the indicated block as output block.

        Arguments:
            block_name: The name of the indicated neural network block.
            upsample_out: The indicator of whether the output should be
                upsampled.
            upsample_mode: The algorithm used for upsampling.
        '''
        if self.cur_block is not None:
            for param in self.cur_block.parameters():
                param.requires_grad = False
        self.cur_block_name = block_name
        self.cur_block = self.blocks[block_name]
        for param in self.cur_block.parameters():
            param.requires_grad = True
        self._cur_block_id = list(self.blocks).index(block_name)
        self._blocks_are_cur = [False] * len(self.blocks)
        cur_block_ids = [self._cur_block_id]
        if self._closed_loop_block_id is not None:
            cur_block_ids.append(self._closed_loop_block_id)
        for block_id in cur_block_ids:
            self._blocks_are_cur[block_id] = True
            for in_id in self._block_input_ids[block_id]:
                new_block_id = in_id - len(self.input_names) - len(
                    self.hidden_names)
                if new_block_id >= 0 and not self._blocks_are_cur[
                    new_block_id]:
                    cur_block_ids.append(new_block_id)
        self.set_upsample(upsample_out, upsample_mode)
        self.train_scale_factor = self.train_scale_factors[self.cur_block_name]
        cur_block_names = [name for name, is_cur in zip(list(self.blocks),
            self._blocks_are_cur) if is_cur]
        self.cur_min_scale_factor = min([self.scale_factors[name] for name in
            cur_block_names] + [self.train_scale_factors[self.cur_block_name]])
        self.get_mask = None
        if hasattr(self.cur_block, 'get_mask'):
            self.get_mask = self.cur_block.get_mask
        self.train(self.training)

    def unset_cur_block(self) -> None:
        '''Unsets the output block (useful for saving and loading lrnn).'''
        if self.cur_block is not None:
            for param in self.cur_block.parameters():
                param.requires_grad = False
        self.cur_block_name = None
        self.cur_block = None
        self.upsample = None
        self.scale_factor = None
        self.train_scale_factor = None
        self.cur_min_scale_factor = None
        self.get_mask = None
        self._cur_block_id = None
        self._blocks_are_cur = [False] * len(self.blocks)

    def set_closed_loop_block(self, block_name: str) -> None:
        '''Sets the indicated block as closed-loop output block.

        Arguments:
            block_name: The name of the indicated neural network block.
        '''
        self.closed_loop_block_name = block_name
        self._closed_loop_block_id = list(self.blocks).index(block_name)
        if self.cur_block_name is not None:
            self.set_cur_block(self.cur_block_name)

    def unset_closed_loop_block(self) -> None:
        '''Unsets the closed-loop output block.'''
        self.closed_loop_block_name = None
        self._closed_loop_block_id = None
        if self.cur_block_name is not None:
            self.set_cur_block(self.cur_block_name)

    def set_upsample(self, upsample_out: bool = None, upsample_mode: str =
        None) -> None:
        '''Sets appropriately the upsample block.

        Arguments:
            upsample_out: The indicator of whether the output should be
                upsampled.
            upsample_mode: The algorithm used for upsampling.
        '''
        if upsample_out is not None:
            self.upsample_out = upsample_out
        if upsample_mode is not None:
            self.upsample_mode = upsample_mode
        if self.upsample_out:
            self.upsample = Upsample(round(1 / self.scale_factors[
                self.cur_block_name]), self.upsample_mode)
            self.scale_factor = 1
        else:
            self.upsample = nn.Identity()
            self.scale_factor = self.scale_factors[self.cur_block_name]

    def train(self, mode: bool = True, upsample_out: bool = None,
        upsample_mode: str = None) -> None:
        '''Sets the neural network to training mode.

        Arguments:
            mode: The indicator of whether to set training mode or evaluation
                mode.
            upsample_out: The indicator of whether the output should be
                upsampled.
            upsample_mode: The algorithm used for upsampling.
        '''
        super(HLRNN, self).train(mode)
        if mode:
            for block in self._blocks:
                block.eval()
            if self.cur_block is not None:
                self.cur_block.train()
        if self.cur_block is not None:
            self.set_upsample(upsample_out, upsample_mode)


def main(dataset_name: str = 'SynthPlanInput') -> None:
    '''Builds and evaluates a stack of LRBs.

    Arguments:
        dataset_name: The name of the PyTorch Dataset.
    '''
    dataset = getattr(inputdata, dataset_name)({'splitprops': [0.6, 0.2, 0.2],
        'splitset': 0})
    dataset.set_classif(dataset.classifs[0])
    #device = "cuda" if torch.cuda.is_available() else "cpu"

    #model = HLRNN(OrderedDict([
    #        ('rec0', ShallowRNN(dataset.size, 64, len(dataset.all_lbls[
    #            dataset.classif]))),
    #        ('lrb1', LRB(dataset.size, (16, 16), 32, 16, 2, 3, 2, 5,
    #            return_contr=False)),
    #        ('rec1', ShallowRNN(64, 64, len(dataset.all_lbls[
    #            dataset.classif]))),
    #        ('lrb2', LRB(64, (16, 16), 32, 16, 2, 5, 3, 5, return_contr=
    #            False)),
    #        ('rec2', ShallowRNN(64, 64, len(dataset.all_lbls[
    #            dataset.classif])))]),
    #    ['input'],
    #    {'rec0': 'input', 'lrb1': 'input', 'rec1': 'lrb1', 'lrb2': 'lrb1',
    #        'rec2': 'lrb2'},
    #    ['rec0_hidden', 'lrb1_rec', 'lrb1_temp', 'rec1_hidden', 'lrb2_rec',
    #        'lrb2_temp', 'rec2_hidden'],
    #    {'rec0': 'rec0_hidden', 'lrb1': ['lrb1_rec', 'lrb1_temp'], 'rec1':
    #        'rec1_hidden', 'lrb2': ['lrb2_rec', 'lrb2_temp'], 'rec2':
    #        'rec2_hidden'})#.to(device)

    model = HLRNN(OrderedDict([
            ('rec0', ShallowRNN(dataset.size, 64, len(dataset.all_lbls[
                dataset.classif]))),
            ('tmpl0', TempMaxPool(3, 2)),
            ('lrb1', NILRNN(dataset.size, (16, 16), 32, 16, 2, 5,
                rec_nonlinearity='relu', train_out_nonlinearity='relu',
                return_contr=False)),
            ('rec1', ShallowRNN(64, 64, len(dataset.all_lbls[
                dataset.classif]))),
            ('tmpl1', TempMaxPool(5, 3)),
            ('lrb2', NILRNN(64, (16, 16), 32, 16, 2, 5, rec_nonlinearity=
                'relu', train_out_nonlinearity='relu', return_contr=False)),
            ('rec2', ShallowRNN(64, 64, len(dataset.all_lbls[
                dataset.classif])))]),
        ['input'],
        {'rec0': 'input', 'tmpl0': 'input', 'lrb1': 'tmpl0', 'rec1': 'lrb1',
            'tmpl1': 'lrb1', 'lrb2': 'tmpl1', 'rec2': 'lrb2'},
        ['rec0_hidden', 'tmpl0_hidden', 'lrb1_hidden', 'rec1_hidden',
            'tmpl1_hidden', 'lrb2_hidden', 'rec2_hidden'],
        {'rec0': 'rec0_hidden', 'tmpl0': 'tmpl0_hidden', 'lrb1': 'lrb1_hidden',
            'rec1': 'rec1_hidden', 'tmpl1': 'tmpl1_hidden', 'lrb2':
            'lrb2_hidden', 'rec2': 'rec2_hidden'})#.to(device)

    def unsup_train(dataloader, model, loss_fn, opt):
        num_batches = 50
        num_samples = num_batches * dataloader.batch_size
        model.train()
        for batch, (*x, _) in enumerate(itertools.islice(dataloader,
            num_batches)):
            #x = tuple(c.to(device) for c in x)
            output = model(*x)
            loss = loss_fn(*output[:len(output) - len(model.hidden_names)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if batch % 10 == 0:
                loss, current = loss.item(), batch * dataloader.batch_size
                print(f'loss: {loss:>7f}  [{current:>5d}/{num_samples:>5d}]')

    def sup_train(dataloader, model, loss_fn, opt):
        num_batches = 50
        num_samples = num_batches * dataloader.batch_size
        model.train()
        for batch, (x, y) in enumerate(itertools.islice(dataloader,
            num_batches)):
            #x, y = x.to(device), y.to(device)
            pred, *_ = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if batch % 10 == 0:
                loss, current = loss.item(), batch * dataloader.batch_size
                print(f'loss: {loss:>7f}  [{current:>5d}/{num_samples:>5d}]')

    def sup_test(dataloader, model, loss_fn):
        num_batches = 100
        num_samples = num_batches * dataloader.batch_size
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in itertools.islice(dataloader, num_batches):
                #x, y = x.to(device), y.to(device)
                pred, *_ = model(x)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= num_samples
        print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: '
              f'{test_loss:>8f}')

    def full_up_unsup(dataset, model):
        block_batch_size = 100
        input_batch_size = round(math.ceil(block_batch_size *
            model.cur_min_scale_factor / model.train_scale_factor) /
            model.cur_min_scale_factor)
        dataloader = DataLoader(dataset, batch_size=input_batch_size)
        lr, lmbd, beta, rho = 1e-3, 1e-5, 1e-3, .02
        mse_loss_fn = MSELoss(model.get_mask)
        weight_decay_loss_fn = WeightDecayLoss(model.cur_block)
        sparsity_loss_fn = SparsityLoss(rho, model.cur_block.sparse_vbles,
            model.cur_block.sparse_nonlinearity)
        loss_fn = lambda pred, y: mse_loss_fn(pred, y) + lmbd * \
            weight_decay_loss_fn() + beta * sparsity_loss_fn()
        opt = torch.optim.Adam(model.cur_block.parameters(), lr=lr)
        epochs = 5
        dataset.restart({'splitset': 0})
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}\n-------------------------------')
            unsup_train(dataloader, model, loss_fn, opt)

    def full_sup(dataset, model):
        block_batch_size = 100
        input_batch_size = round(math.ceil(block_batch_size *
            model.cur_min_scale_factor / model.train_scale_factor) /
            model.cur_min_scale_factor)
        dataloader = DataLoader(dataset, batch_size=input_batch_size)
        lr, lmbd = 1e-3, 1e-5
        cross_entropy_loss_fn = nn.CrossEntropyLoss()
        weight_decay_loss_fn = WeightDecayLoss(model.cur_block)
        loss_fn = lambda pred, y: cross_entropy_loss_fn(pred, y) + lmbd * \
            weight_decay_loss_fn()
        opt = torch.optim.Adam(model.cur_block.parameters(), lr=lr)
        epochs = 5
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}\n-------------------------------')
            dataset.restart({'splitset': 0})
            sup_train(dataloader, model, loss_fn, opt)
            dataset.restart({'splitset': 1})
            sup_test(dataloader, model, loss_fn)

    print('\n\nBlock rec0:\n-------------------------------')
    model.set_cur_block('rec0', True)
    full_sup(dataset, model)
    print('\n\nBlock lrb1:\n-------------------------------')
    model.set_cur_block('lrb1', False)
    full_up_unsup(dataset, model)
    print('\n\nBlock rec1:\n-------------------------------')
    model.set_cur_block('rec1', True)
    full_sup(dataset, model)
    print('\n\nBlock lrb2:\n-------------------------------')
    model.set_cur_block('lrb2', False)
    full_up_unsup(dataset, model)
    print('\n\nBlock rec2:\n-------------------------------\n')
    model.set_cur_block('rec2', True)
    full_sup(dataset, model)

    model.extend(OrderedDict([
            ('tmpl2', TempMaxPool(3, 2)),
            ('lrbpred2', GRULRBPredictor(64, 4, 56, 56)),
            ('upsample2', MultiUpsample(2)),
            #('trim2', TrimLike()),
            ('lrbpred1', LRBPredictor(64, 2, 56, 64, 4)),
            ('upsample1', MultiUpsample(3)),
            #('trim1', TrimLike()),
            ('lrbpred0', LRBPredictor(dataset.size, 3, 56, 64, 2))]),
        {'tmpl2': 'lrb2', 'lrbpred2': 'tmpl2', 'upsample2': ['tmpl2',
            'lrbpred2'], 'lrbpred1': ['tmpl1', 'upsample2'], 'upsample1': [
            'tmpl1', 'lrbpred1'], 'lrbpred0': ['tmpl0', 'upsample1']},
        ['tmpl2_hidden', 'lrbpred2_hidden'],
        {'tmpl2': 'tmpl2_hidden', 'lrbpred2': 'lrbpred2_hidden'})

    def full_down_unsup(dataset, model):
        block_batch_size = 100
        input_batch_size = round(math.ceil(block_batch_size *
            model.cur_min_scale_factor / model.train_scale_factor) /
            model.cur_min_scale_factor)
        dataloader = DataLoader(dataset, batch_size=input_batch_size)
        lr, lmbd = 1e-3, 1e-5
        nll_loss_fn = RectGaussMixNLLLoss(model.get_mask)
        weight_decay_loss_fn = WeightDecayLoss(model.cur_block)
        loss_fn = lambda mean, var, coef, y: nll_loss_fn(mean, var, coef, y) \
            + lmbd * weight_decay_loss_fn()
        opt = torch.optim.Adam(model.cur_block.parameters(), lr=lr)
        epochs = 5
        dataset.restart({'splitset': 0})
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}\n-------------------------------')
            unsup_train(dataloader, model, loss_fn, opt)

    print('\n\nBlock lrbpred2:\n-------------------------------')
    model.set_cur_block('lrbpred2', False)
    full_down_unsup(dataset, model)
    print('\n\nBlock lrbpred1:\n-------------------------------')
    model.set_cur_block('lrbpred1', False)
    full_down_unsup(dataset, model)
    print('\n\nBlock lrbpred0:\n-------------------------------')
    model.set_cur_block('lrbpred0', False)
    full_down_unsup(dataset, model)

    model.extend(OrderedDict([
            ('rec1p', ShallowRNN(64, 64, len(dataset.all_lbls[
                dataset.classif]))),
            ('samp1', RectGaussMix())]),
        {'rec1p': 'tmpl1', 'samp1': 'lrbpred1'},
        ['rec1p_hidden'],
        {'rec1p': 'rec1p_hidden'})

    def pred_test(dataloader, model, closed_loop_block_name, horizon):
        block_batch_size = 100
        input_batch_size = round(math.ceil(block_batch_size *
            model.cur_min_scale_factor / model.train_scale_factor) /
            model.cur_min_scale_factor)
        dataloader = DataLoader(dataset, batch_size=input_batch_size)
        mc_samples = 8
        num_batches = 10
        num_samples = num_batches * input_batch_size - horizon
        in_block_name = model.block_inputs[model.cur_block_name]
        submodel = model.get_sub_hlrnn([model.cur_block_name,
            closed_loop_block_name], [in_block_name])
        submodel.set_cur_block(model.cur_block_name, False)
        submodel.set_closed_loop_block(closed_loop_block_name)
        model.set_cur_block(in_block_name, False)
        closed_loop_horizon = math.ceil(horizon * model.scale_factor) + 1
        floor_horizon = math.floor(horizon * model.scale_factor)
        closed_loop_stride = round(1 / model.scale_factor)
        pred_shift = horizon % closed_loop_stride
        model.eval()
        submodel.eval()
        preds = torch.tensor([])
        correct, candidate, n_candidates = 0, 0, 0
        with torch.no_grad():
            for x, y in itertools.islice(dataloader, num_batches):
                #x, y = x.to(device), y.to(device)
                subinput, *_ = model(x)
                pred, *_ = submodel.closed_loop_forward(subinput, horizon=
                    closed_loop_horizon, repetitions=mc_samples)
                pred = pred[:, :, [floor_horizon] * (closed_loop_stride -
                    pred_shift) + [floor_horizon + 1] * pred_shift].permute(0,
                    2, 1, 3).reshape(-1, mc_samples, pred.shape[-1])
                preds = torch.cat([preds, pred], 0)
                if preds.shape[0] > horizon:
                    pred = preds[:-horizon]
                    preds = preds[-horizon:]
                    if y.shape[0] > pred.shape[0]:
                        y = y[-pred.shape[0]:]
                    ys_ = pred.argmax(2)
                    correct += (ys_.mode(1)[0] == y).type(
                        torch.float).sum().item()
                    candidate += (ys_ == y.unsqueeze(1)).any(1).type(
                        torch.float).sum().item()
                    n_candidates += (ys_.sort(1)[0].diff(dim=1) != 0).type(
                        torch.float).sum().item()
        correct /= num_samples
        candidate /= num_samples
        n_candidates = n_candidates / num_samples + 1
        print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Candidate: '
              f'{(100*candidate):>0.1f}%, Avg num candidates: '
              f'{(n_candidates):>0.2f}')

    print('\n\nBlock rec1p:\n-------------------------------')
    model.set_cur_block('rec1p', True)
    full_sup(dataset, model)
    print('\n\nTest prediction:\n-------------------------------')
    pred_test(dataset, model, 'samp1', 15)

    classif = dataset.classifs[0]
    if dataset_name == 'SynthPlanInput':
        classif = 'plans'
    class LabelAtInputDataset(getattr(inputdata, dataset_name)):
        def __init__(self, params):
            super().__init__(params)
            self.set_classif(classif)
        def __next__(self):
            sample, label = super().__next__()
            return sample, F.one_hot(label, len(self.all_lbls[
                self.classif])), torch.tensor([], dtype=torch.int64)
    inputdata._matlab_eng = None
    dataset = LabelAtInputDataset({'splitprops': [0.6, 0.2, 0.2], 'splitset':
        0})

    model.extend_input(['label'])
    model.extend(OrderedDict([
            ('downsample', Downsample(12)),
            ('lrbpred2p', GRULRBPredictor(64, 4, 56, 56, len(dataset.all_lbls[
                dataset.classif]))),
            ('peak2', RectGaussMixPeak()),
            ('upsample2p', MultiUpsample(2)),
            #('trim2', TrimLike()),
            ('lrbpred1p', LRBPredictor(64, 2, 56, 64, None, 2)),
            ('peak1', RectGaussMixPeak()),
            ('upsample1p', MultiUpsample(3)),
            #('trim1', TrimLike()),
            ('lrbpred0p', LRBPredictor(dataset.size, 3, 56, 64, None, 3)),
            ('peak0', RectGaussMixPeak()),
            ('upsample0p', Upsample(2))]),
        {'downsample': 'label', 'lrbpred2p': ['tmpl2', 'downsample'], 'peak2':
            'lrbpred2', 'upsample2p': ['tmpl2', 'peak2'], 'lrbpred1p': [
            'tmpl1', 'upsample2p'], 'peak1': 'lrbpred1p', 'upsample1p': [
            'tmpl1', 'peak1'], 'lrbpred0p': ['tmpl0', 'upsample1p'], 'peak0':
            'lrbpred0p', 'upsample0p': 'peak0'},
        ['lrbpred2p_hidden'],
        {'lrbpred2p': 'lrbpred2p_hidden'})

    print('\n\nBlock lrbpred2p:\n-------------------------------')
    model.set_cur_block('lrbpred2p', False)
    full_down_unsup(dataset, model)
    print('\n\nBlock lrbpred1p:\n-------------------------------')
    model.set_cur_block('lrbpred1p', False)
    full_down_unsup(dataset, model)
    print('\n\nBlock lrbpred0p:\n-------------------------------')
    model.set_cur_block('lrbpred0p', False)
    full_down_unsup(dataset, model)

    def action_selection_test(dataset, model):
        config = {'num_trials': 16, 'num_samples_factor': 2}
        for block in model.blocks.values():
            if type(block) == LRBPredictor:
                block.use_actual_ctxt = False
        achieved, avg_num_samples = actionselection.test(config, dataset,
            model)
        avg_num_samples = 'unknown' if avg_num_samples is None else \
            f'{(avg_num_samples):>0.2f}'
        print(f'Test Error: \n Achieved: {(100*achieved):>0.1f}%, Avg num '
              f'samples: {avg_num_samples}')

    def show_action_selection(dataset, model):
        config = {'num_trials': 256, 'num_samples_factor': 2}
        env = actionselection.SynthActionsEnvironment(dataset)
        scenarioplot = plt.imshow(env.get_scenario_plot())
        def plot_fn(env, goal, _):
            scenarioplot.set_data(env.get_scenario_plot())
            plt.title(f'goal: {goal}')
            plt.show(block=False)
            plt.pause(0.033)
        actionselection.test(config, dataset, model, plot_fn)
        

    if dataset_name == 'SynthPlanInput':
        print('\n\nTest action selection:\n-------------------------------')
        model.set_cur_block('upsample0p')
        model.set_closed_loop_block('upsample0p')
        action_selection_test(dataset, model)
        show_action_selection(dataset, model)
    

    print("\nDone!")


if __name__ == '__main__':
    import actionselection
    from torch.utils.data import DataLoader
    import inputdata
    import itertools
    import sys
    import matplotlib.pyplot as plt

    main(*sys.argv[1:])
