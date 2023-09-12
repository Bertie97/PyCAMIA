#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import os, sys
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torchext import Optim, torchext

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

@torchext
def nbatch(tensor): return tensor.size(0)
@torchext
def cat(*tensors): return torch.cat(tensors, 1)
def parse(string):
    if string.count('(') > 1 or string.count(')') > 1: raise TypeError("Invalid to parse: " + string + ". ")
    if string.count('(') == 0 and string.count(')') == 0: string += '()'
    return eval('("' + string.lower().replace('(', '", (').replace(')', ',)').replace('(,)', '()') + ')')
@torchext
def combine(list_of_items, reduction):
    if len(list_of_items) >= 2:
        z = reduction(list_of_items[0], list_of_items[1])
        for i in range(2, len(list_of_items)):
            z = reduction(z, list_of_items[i])
    else: z = list_of_items[0]
    return z

@torchext
def crop_as(x, y, center='center', keepdims=0):
    if isinstance(x, np.ndarray): size_x = x.shape
    elif isinstance(x, torch.Tensor): size_x = x.size()
    else: raise TypeError("Unknown type for x in crop_as. ")
    if isinstance(y, np.ndarray): size_y = y.shape
    elif isinstance(y, torch.Tensor): size_y = y.size()
    elif isinstance(y, tuple): size_y = y
    else: raise TypeError("Unknown type for y in crop_as. ")
    size_y = size_x[:keepdims] + size_y[keepdims:]
    if all([a == b for a, b in zip(size_x, size_y)]) and center == 'center': return x
    if center == 'center': center = tuple(m / 2 for m in size_x)
    center = tuple(m / 2 for m in size_x)[:len(size_x) - len(center)] + center
    assert len(size_x) == len(size_y) == len(center)
    if isinstance(x, np.ndarray): z = np.zeros(size_y)
    else: z = torch.zeros(*size_y).to(device)
    intersect = lambda a, b: (max(a[0], b[0]), min(a[1], b[1]))
    z_box = [intersect((0, ly), (- int(m - ly / 2), - int(m - ly / 2) + lx)) for m, lx, ly in zip(center, size_x, size_y)]
    x_box = [intersect((0, lx), (+ int(m - ly / 2), + int(m - ly / 2) + ly)) for m, lx, ly in zip(center, size_x, size_y)]
    if any([r[0] >= r[1] for r in z_box]) or any([r[0] >= r[1] for r in x_box]): return z
    region_z = torch.meshgrid(*[torch.arange(*r).to(device) for r in z_box])
    region_x = torch.meshgrid(*[torch.arange(*r).to(device) for r in x_box])
    z[region_z] = x[region_x]
    return z
    
#### test ####
#x = torch.rand(437, 521, 733)
#y = torch.rand(3, 3, 3)
#print(x[217:220, 259:262, 365:368])
#print(crop_as(x, y))
##############


class Convolution_Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, **params):
        '''
        ::parameters:
            dimension (int): The dimension of the images. 
            in_channels (int): The input channels for the block. 
            out_channels (int): The output channels for the block. 
            conv_num (int): The number of convolution layers. 
            kernel_size (int): The size of the convolution kernels. 
            padding (int): The image padding for the convolutions. 
            activation_function (class): The activation function. 
            active_args (dict): The arguments for the activation function. 
            conv_block (str): A string with possible values in ('conv', 'dense', 'residual'), indicating which kind of block the U-Net is using: normal convolution layers, DenseBlock or ResidualBlock. 
            res_type (function): The combining type for the residual connections.
        '''
        super().__init__()
        default_values = {'dimension': 2, 'conv_num': 1, 'padding': 1, 'kernel_size': 3, 'conv_block': 'conv', 'res_type': torch.add, 'activation_function': nn.ReLU, 'active_args': {}}
        param_values = {}
        param_values.update(default_values)
        param_values.update(params)
        self.__dict__.update(param_values)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.layers = nn.ModuleList()
        for i in range(self.conv_num):
            ic = self.in_channels if i == 0 else ((self.out_channels * i + self.in_channels) if self.conv_block == 'dense' else self.out_channels)
            conv = eval('nn.Conv%dd' % self.dimension)(ic, self.out_channels, self.kernel_size, 1, self.padding)
            initialize_model, initialize_params = parse(self.initializer)
            eval('nn.init.%s_' % initialize_model)(conv.weight, *initialize_params)
            if self.conv_block != 'dense': self.layers.append(conv)
            oc = (self.out_channels * i + self.in_channels) if self.conv_block == 'dense' else self.out_channels 
            self.layers.append(eval('nn.BatchNorm%dd' % self.dimension)(oc))
            if i < self.conv_num: self.layers.append(self.activation_function(**self.active_args))
            if self.conv_block == 'dense': self.layers.append(conv)
    
    @torchext
    def forward(self, x):
        if self.conv_block == 'dense':
            conv_results = [x]
            conv_layer = True
            for layer in self.layers:
                if conv_layer: x = layer(torch.cat([crop_as(l, conv_results[-1], keepdims=2) for l in conv_results], 1))
                else: x = layer(x)
                conv_layer = layer.__class__.__name__.startswith('Conv')
                if conv_layer: conv_results.append(x)
            return self.activation_function(**self.active_args)(x)
        else:
            y = x
            for layer in self.layers: y = layer(y)
            if self.conv_block == 'residual': z = self.res_type(crop_as(x, y, keepdims=2), y)
            else: z = y
            return self.activation_function(**self.active_args)(z)
        

class U_Net(nn.Module):
    
    class Softmax(nn.Module):
        @torchext
        def forward(self, x): return nn.functional.softmax(x, 1)
    
    class Encoder_Block(nn.Module):
        
        def __init__(self, in_channels, out_channels, has_pooling, params):
            super().__init__()
            block_params = params.copy()
            block_params.update({'in_channels': in_channels, 'out_channels': out_channels, 'has_pooling': has_pooling})
            self.__dict__.update(block_params)
            if has_pooling: self.pooling = eval('nn.MaxPool%dd' % self.dimension)(self.pooling_size, ceil_mode = self.padding == self.kernel_size // 2)
            self.conv_block = Convolution_Block(**block_params)
        
        @torchext
        def forward(self, x):
            if self.has_pooling: y = self.pooling(x)
            else: y = x
            return self.conv_block(y)
            
            
    class Decoder_Block(nn.Module):
        
        def __init__(self, list_of_encoders, in_channels, out_channels, params, copies_of_inputs):
            super().__init__()
            block_params = params.copy()
            block_params.update({'in_channels': in_channels, 'out_channels': out_channels})
            self.__dict__.update(block_params)
            if self.skip_type == cat: to_channels = in_channels - list_of_encoders[0].out_channels
            else: assert all([in_channels == encoder.out_channels for encoder in list_of_encoders]); to_channels = in_channels
            self.upsampling = eval('nn.ConvTranspose%dd' % self.dimension)(in_channels * copies_of_inputs, to_channels, self.pooling_size, self.pooling_size, 0)
            block_params.update({'in_channels': to_channels + sum([encoder.out_channels for encoder in list_of_encoders]), 'out_channels': out_channels})
            self.conv_block = Convolution_Block(**block_params)
        
        @torchext
        def forward(self, x, list_of_encoder_results):
            y = self.upsampling(x)
            if self.padding == self.kernel_size // 2:
                to_combine = list_of_encoder_results + [crop_as(y, list_of_encoder_results[0], keepdims=2)]
            else: to_combine = [crop_as(encoder_result, y, keepdims=2) for encoder_result in list_of_encoder_results] + [y]
            joint = combine(to_combine, self.skip_type)
            return self.conv_block(joint)


    def __init__(self, **params):
        '''
        ::paramerters:
            dimension (int): The dimension of the images. For conventional U-Net, it is 2. 
            depth (int): The depth of the U-Net. The conventional U-Net has a depth of 4 there are 4 pooling layers and 4 up-sampling layers.
            conv_num (int): The number of continuous convolutions in one block. In a conventional U-Net, this is 2. 
            padding (int or str): Indicate the type of padding used. In a conventional U-Net, padding should be 0 but yet the default value is 'SAME' here. 
            in_channels (int): The number of channels for the input. In a conventional U-Net, it should be 1.
            out_channels (int): The number of channels for the output. In a conventional U-Net, it should be 2.
            block_channels (int): The number of channels for the first block if a number is provided. In a conventional U-Net, it should be 64. If a list is provided, the length should be the same as the number of blocks plus two (2 * depth + 3). It represents the channels before and after each block (with the output channels included). Or else, a function may be provided to compute the output channels given the block index (-1 ~ 2 * depth + 1). 
            kernel_size (int): The size of the convolution kernels. In a conventional U-Net, it should be 3. 
            pooling_size (int): The size of the pooling kernels. In a conventional U-Net, it should be 2. 
            // keep_prob (float): The keep probability for the dropout layers. 
            conv_block (str): A string with possible values in ('conv', 'dense', 'residual'), indicating which kind of block the U-Net is using: normal convolution layers, DenseBlock or ResidualBlock. 
            multi_arms (str): A string with possible values in ('shared(2)', 'seperate(2)'), indicating which kind of encoder arms are used. 
            multi_arms_combine (function): The combining type for multi-arms. See skip_type for details. 
            skip_type (function): The skip type for the skip connections. The conventional U-Net has a skip connect of catenation (cat). Other possible skip types include torch.mul or torch.add. 
            res_type (function): The combining type for the residual connections. It should be torch.add in most occasions. 
            activation_function (class): The activation function used after the convolution layers. nn.ReLU by default. 
            active_args (dict): The arguments for the activation function. {} by default. 
            initializer (str): A string indicating the initialing strategy. Possible values are normal(0, 0.1) or uniform(-0.1, 0.1) or constant(0) (all parameters can be changed)
            with_softmax (bool): Whether a softmax layer is applied at the end of the network. 
        '''
        super().__init__()
        default_values = {'dimension': 2, 'depth': 4, 'conv_num': 2, 'padding': 'SAME', 'in_channels': 1, 'out_channels': 2, 'block_channels': 64, 'kernel_size': 3, 'pooling_size': 2, 'keep_prob': 0.5, 'conv_block': 'conv', 'multi_arms': "shared", 'multi_arms_combine': cat, 'skip_type': cat, 'res_type': torch.add, 'activation_function': nn.ReLU, 'active_args': {}, 'initializer': "normal(0, 0.1)", 'with_softmax': True}
        param_values = {}
        param_values.update(default_values)
        param_values.update(params)
        self.__dict__.update(param_values)
        
        if isinstance(self.block_channels, int): self.block_channels = [self.in_channels] + [self.block_channels << min(i, 2 * self.depth - i) for i in range(2 * self.depth + 1)] + [self.out_channels]
        bchannels = self.block_channels
        if not callable(self.block_channels): self.block_channels = lambda i: bchannels[i + 1]
        
        if isinstance(self.padding, str): self.padding = {'SAME': self.kernel_size // 2, 'ZERO': 0, 'VALID': 0}.get(self.padding.upper(), self.kernel_size // 2)
        
        param_values = {k: self.__dict__[k] for k in param_values}
        
        self.arm_type, self.arm_num = parse(self.multi_arms)
        self.arm_num = 1 if len(self.arm_num) == 0 else self.arm_num[0]
        if self.arm_type == 'shared': self.dif_arm_num = 1
        else: self.dif_arm_num = self.arm_num
        
        for iarm in range(self.dif_arm_num):
            for k in range(self.depth + 1):
                setattr(self, 'block%d_%d' % (k, iarm), self.Encoder_Block(self.block_channels(k - 1), self.block_channels(k), k != 0, param_values))
                
        for k in range(self.depth + 1, 2 * self.depth + 1):
            setattr(self, 'block%d' % k, self.Decoder_Block(
                [getattr(self, 'block%d_%d' % (2 * self.depth - k, iarm)) for iarm in range(self.dif_arm_num)] * (self.arm_num // self.dif_arm_num), 
                self.block_channels(k - 1), self.block_channels(k), param_values, 
                self.arm_num if k == self.depth + 1 and self.multi_arms_combine == cat else 1
            ))
            
        conv = eval('nn.Conv%dd' % self.dimension)(self.block_channels(2 * self.depth), self.block_channels(2 * self.depth + 1), 1)
        initialize_model, initialize_params = parse(self.initializer)
        eval('nn.init.%s_' % initialize_model)(conv.weight, *initialize_params)
        if self.with_softmax: self.output_layer = nn.Sequential(conv, self.Softmax())
        else: self.output_layer = conv

    @torchext
    def forward(self, x):
        size = x.size()[1:]
        if len(size) == self.dimension and self.in_channels == 1: x = x.unsqueeze(1)
        elif len(size) == self.dimension + 1 and self.in_channels * self.arm_num == size[0]: pass
        else: raise ValueError("The input tensor does not correspond to the U-Net structure. ")
        
        assert size[0] % self.arm_num == 0
        inputs = x.split(size[0] // self.arm_num, 1)
        assert len(inputs) == self.arm_num
        
        for i, y in enumerate(inputs):
            for k in range(self.depth + 1):
                y = getattr(self, 'block%d_%d' % (k, 0 if self.arm_type == 'shared' else i))(y)
                setattr(self, 'block%d_%dresult' % (k, i), y)
                
        to_combine = [getattr(self, 'block%d_%dresult' % (self.depth, i)) for i in range(self.arm_num)]
        z = combine(to_combine, self.multi_arms_combine)
            
        for k in range(self.depth + 1, 2 * self.depth + 1):
            z = getattr(self, 'block%d' % k)(z, [getattr(self, 'block%d_%dresult' % (2 * self.depth - k, iarm)) for iarm in range(self.arm_num)])
            setattr(self, 'block%dresult' % k, z)
            
        return self.output_layer(z)
        
    def optimizer(self, lr=0.001): return Optim(torch.optim.Adam, self.parameters(), lr)

    @torchext
    def loss(self, x, y):
        y_hat = self(x)
        clamped = y_hat.clamp(1e-10, 1.0)
        return y_hat, - torch.sum(y * torch.log(clamped), 1).mean()
        
    def __getitem__(self, i):
        if self.arm_num == 1 and i <= self.depth: i = (i, 0)
        return getattr(self, 'block%dresult' % i if isinstance(i, int) else 'block%d_%dresult' % i)
        
    def __iter__(self):
        for i in range(2 * self.depth + 1):
            if i <= self.depth:
                for iarm in range(self.arm_num):
                    yield 'block%d_%dresult' % (i, iarm), (i, iarm)
            else: yield 'block%dresult' % i, i
        
if __name__ == "__main__":
#    unet = U_Net(multi_arms="seperate(3)", block_channels=16)
#    print(unet(torch.rand(10, 3, 100, 100)).size())
#    print(*[x + ' ' + str(unet[i].size()) for x, i in unet], sep='\n')
    unet = U_Net(
        dimension=3, 
        in_channels=2, 
        out_channels=3, 
        block_channels=4, 
        with_softmax=False, 
        initializer="normal(0.0, 0.9)", 
#        conv_block='dense', 
#        conv_num=4, 
#        active_args={'inplace': True}
    )
    print(unet(torch.rand(10, 2, 50, 50, 50)).size())
    print(*[x + ' ' + str(unet[i].size()) for x, i in unet], sep='\n')