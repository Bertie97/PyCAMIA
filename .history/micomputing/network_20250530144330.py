
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "micomputing",
    author = "Yuncheng Zhou",
    create = "2022-03",
    fileinfo = "File containing U-net and convolutional networks.",
    help = "Use `from micomputing import *`.",
    requires = "batorch"
).check()

__all__ = """
    U_Net
    CNN
    FCN
    RNN
    NeuralODE
    NeuralODEMarch
    CoupledNeuralODEMarch
    Convolution_Block
    Convolution
    Models
""".split()
    
import math

with __info__:
    import batorch as bt
    from batorch import nn
    from pycamia import touch, avouch, execblock
    from pycamia import Path, ByteSize, SPrint, tokenize, get_environ_vars

def parse(string):
    if string.count('(') > 1 or string.count(')') > 1: raise TypeError("Invalid to parse: " + string + ". ")
    if string.count('(') == 0 and string.count(')') == 0: string += '()'
    return eval('("' + string.lower().replace('(', '", (').replace(')', ',)').replace('(,)', '()') + ')')

def parse_activations(act, finalact, vars):
    if act is None: act = NoAct()
    if isinstance(act, type): act = act()
    if isinstance(act, str):
        if '(' not in act: act += '()'
        act = eval(act, vars.locals, vars.globals)
    if finalact == ...: finalact = act
    if finalact is None: finalact = NoAct()
    if isinstance(finalact, type): finalact = finalact()
    if isinstance(finalact, str):
        if '(' not in finalact: finalact += '()'
        finalact = eval(finalact, vars.locals, vars.globals)
    return act, finalact

def combine(list_of_items, reduction):
    if len(list_of_items) >= 2:
        z = reduction(list_of_items[0], list_of_items[1])
        for i in range(2, len(list_of_items)):
            z = reduction(z, list_of_items[i])
    else: z = list_of_items[0]
    return z

Convolution = {
    0: lambda ic, oc, *_: nn.Linear(ic, oc), 
    1: nn.Conv1d, 
    2: nn.Conv2d, 
    3: nn.Conv3d
}

MaxPool = {
    0: lambda *_: (lambda x: x), 
    1: nn.MaxPool1d, 
    2: nn.MaxPool2d, 
    3: nn.MaxPool3d
}

ConvTranspose = {
    0: lambda *_: (lambda x: x), 
    1: nn.ConvTranspose1d, 
    2: nn.ConvTranspose2d, 
    3: nn.ConvTranspose3d
}

class BatchNorm(nn.Module):
    def __init__(self, idim, ch):
        super().__init__()
        self.idim = idim
        if idim > 0: self.flattenedBatchNorm = nn.BatchNorm1d(ch)
        
    def __class_getitem__(cls, idim):
        if idim in [1, 2, 3]: return getattr(nn, f"BatchNorm{idim}d")
        return lambda ch: cls(idim, ch)
    
    def forward(self, x):
        if self.idim == 0: return x
        else: return self.flattenedBatchNorm(x.flatten(...))

class Softmax(nn.Module):
    def forward(self, x): return nn.functional.softmax(x, 1)

class NoAct(nn.Module):
    def forward(self, x): return x

class Convolution_Block(nn.Module):
    '''
    Args:
        dimension (int): The dimension of the images. Defaults to 2. 
        in_channels (int): The input channels for the block. 
        out_channels (int): The output channels for the block. 
        mid_channels (int): The middle channels for the block. 
        conv_num (int): The number of convolution layers. Defaults to 1. 
        kernel_size (int): The size of the convolution kernels. Defaults to 3. 
        padding (int): The image padding for the convolutions. Defaults to 1. 
        initializer (str): A string indicating the initialing strategy. Possible values 
            are normal(0, 0.1) or uniform(-0.1, 0.1) or constant(0) (all parameters can be changed)
        linear_layer (bool): Whether the convolutions are used for 1x1 images (equivalent to linear layers). 
        activation (class or str): The activation function. 
        final_activation (class or str): The activation function after the final convolution. 
        conv_block (str): A string with possible values in ('conv', 'dense', 'residual'), 
            indicating which kind of block the U-Net is using: normal convolution layers, 
            DenseBlock or ResidualBlock. Defaults to 'conv'.
        res_type (function): The combining type for the residual connections.
    '''
    
    def __init__(self, in_channels, out_channels, mid_channels=None, **params):
        super().__init__()
        default_values = {'dimension': 2, 'conv_num': 1, 'kernel_size': 3, 'padding': 1, 'linear_layer': False, 'initializer': "normal(0, 0.1)", 'conv_block': 'conv', 'res_type': bt.add, 'activation': nn.ReLU, 'final_activation': ...}
        param_values = {}
        param_values.update(default_values)
        param_values.update(params)
        for k, v in param_values.items(): setattr(self, k, v)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_channels is None:
            if self.linear_layer: self.mid_channels = 5 * self.out_channels
            else: self.mid_channels = self.out_channels
        else:
            self.mid_channels = mid_channels
            
        if self.linear_layer: self.kernel_size = 1; self.padding = 0
        
        if isinstance(self.padding, str): self.padding = {'SAME': self.kernel_size // 2, 'ZERO': 0, 'VALID': 0}.get(self.padding.upper(), self.kernel_size // 2)
        self.activation, self.final_activation = parse_activations(self.activation, self.final_activation, get_environ_vars())
        
        self.layers = nn.ModuleList()
        for i in range(self.conv_num):
            ic = self.in_channels if i == 0 else ((self.mid_channels * i + self.in_channels) if self.conv_block == 'dense' else self.mid_channels)
            oc = self.out_channels if i == self.conv_num - 1 else self.mid_channels
            conv = Convolution[self.dimension](ic, oc, self.kernel_size, 1, self.padding)
            initialize_model, initialize_params = parse(self.initializer)
            eval('nn.init.%s_' % initialize_model)(conv.weight, *initialize_params)
            if self.conv_block != 'dense': self.layers.append(conv)
            oc = (self.mid_channels * i + self.in_channels) if self.conv_block == 'dense' else oc
            if not self.linear_layer: self.layers.append(BatchNorm[self.dimension](oc))
            if i < self.conv_num-1: self.layers.append(self.activation)
            if self.conv_block == 'dense': self.layers.append(conv)

    def forward(self, x):
        self.memory_used = ByteSize(0)
        need_squeeze = False
        original_shape = x.shape
        if self.linear_layer and x.n_space_dim == 0:
            need_squeeze = True
            x = x.view(*x.shape.python_repr, *((1,) * self.dimension))
        if self.conv_block == 'dense':
            conv_results = [x]
            conv_layer = True
            for layer in self.layers:
                try:
                    if conv_layer: x = layer(bt.cat([bt.crop_as(l, conv_results[-1]) for l in conv_results], 1))
                    else: x = layer(x)
                except Exception as e:
                    raise e.__class__(f"In layer {layer}. " + e.__str__())
                self.memory_used += x.byte_size()
                conv_layer = layer.__class__.__name__.startswith('Conv')
                if conv_layer: conv_results.append(x)
            result = self.final_activation(x)
        else:
            y = x
            for layer in self.layers:
                try: y = layer(y)
                except Exception as e:
                    raise e.__class__(f"In layer {layer}. " + e.__str__())
                # print(layer, y)
                self.memory_used += ByteSize(y.numel() * y.element_size())
            y = y.as_subclass(bt.Tensor).special_from(x)
            if self.conv_block == 'residual': z = self.res_type(bt.crop_as(x, y), y)
            else: z = y
            result = self.final_activation(z)
        result = result.as_subclass(bt.Tensor).special_from(x)
        if need_squeeze: result = result.view(original_shape.with_feature(result.feature))
        elif self.dimension > 0 and self.padding == self.kernel_size // 2:
            return bt.crop_as(result, x.space)
        return result

class U_Net(nn.Module):
    '''
    Args:
        variational (bool): Whether the U-Net is variational at the bottleneck and skip connections.
        dimension (int): The dimension of the images. Defaults to 2 (see U-Net). 
        depth (int): The depth of the U-Net. Defaults to 4 indicating 4 pooling layers and 4 up-sampling layers (see U-Net).
        conv_num (int): The number of continuous convolutions in one block. Defaults to 2. 
        padding (int or str): Indicate the type of padding used. Defaults to 'SAME' though it is 0 in conventional U-Net. 
        in_channels (int): The number of channels for the input. Defaults to 1 (see U-Net).
        out_channels (int): The number of channels for the output. Defaults to 2 (see U-Net).
        block_channels (int): The number of channels for the first block if a number is provided. Defaults to 64 (see U-Net). 
            If a list is provided, the length should be the same as the number of blocks plus two (2 * depth + 3). It represents the channels before and after each block (with the output channels included). 
            Or else, a function may be provided to compute the output channels given the block index (-1 ~ 2 * depth + 1) [including input_channels at -1 and output_channels at 2 * depth + 1]. 
        bottleneck_in_channels (int): The number of channels for the bottleneck input. Defaults to 0. 
        bottleneck_out_channels (int): The number of channels for the bottleneck output. Defaults to 0. 
        kernel_size (int): The size of the convolution kernels. Defaults to 3 (see U-Net). 
        pooling_size (int): The size of the pooling kernels. Defaults to 2 (see U-Net). 
        // keep_prob (float): The keep probability for the dropout layers. 
        conv_block (str): A string with possible values in ('conv', 'dense', 'residual'), indicating which kind of block the U-Net is using: normal convolution layers, DenseBlock or ResidualBlock. 
        multi_arms (str): A string with possible values in ('shared(2)', 'seperate(2)'), indicating which kind of encoder arms are used. 
        multi_arms_combine (function): The combining type for multi-arms. See skip_combine for details. 
        skip_combine (function): The skip type for the skip connections. Defaults to catenation (U_Net.cat; see U-Net). Other possible skip types include torch.mul or torch.add. 
        res_type (function): The combining type for the residual connections. It should be torch.add in most occasions. 
        activation (class or str): The activation function used after the convolution layers. Defaults to nn.ReLU. 
        final_activation (class or str): The activation function after the final convolution. 
        initializer (str): A string indicating the initialing strategy. Possible values are normal(0, 0.1) or uniform(-0.1, 0.1) or constant(0) (all parameters can be changed)
        cum_layers (list): A list consisting two numbers [n, m] indicating that the result would be a summation of the upsamples of the results of the nth to the mth (included) blocks, block_numbers are in range 0 ~ 2 * depth. 
            The negative indices are allowed to indicate the blocks in a inversed order with -1 representing the output for the last block. 
    '''
    
    @staticmethod
    def variational_resample(mlv):
        dim = mlv.i_chan_dim
        n_channel = mlv.n_channel // 2
        m_indices = (slice(None),) * dim + (slice(None, n_channel),)
        lv_indices = (slice(None),) * dim + (slice(n_channel, None),)
        mean = mlv[m_indices]
        logvar = mlv[lv_indices]
        return mean + bt.exp(logvar/2) * bt.randn_like(mean)
    
    @staticmethod
    def cat(*tensors): return bt.cat(tensors, 1)
    
    @staticmethod
    def noskip(*tensors): return tensors[-1]
    
    class Encoder_Block(nn.Module):
        
        def __init__(self, in_channels, out_channels, has_pooling, params):
            super().__init__()
            block_params = params.copy()
            block_params.update({'in_channels': in_channels, 'out_channels': out_channels, 'has_pooling': has_pooling})
            for k, v in block_params.items(): setattr(self, k, v)
            if has_pooling: self.pooling = MaxPool[self.dimension](self.pooling_size, ceil_mode = True)
            self.conv_block = Convolution_Block(**block_params)

        def forward(self, x):
            if self.has_pooling: y = self.pooling(x)
            else: y = x
            return self.conv_block(y)
            
    class Decoder_Block(nn.Module):
        
        def __init__(self, list_of_encoders, in_channels, out_channels, params, copies_of_inputs):
            super().__init__()
            block_params = params.copy()
            block_params.update({'in_channels': in_channels, 'out_channels': out_channels})
            for k, v in block_params.items(): setattr(self, k, v)
            
            arm_channels = list_of_encoders[0].out_channels
            if self.multi_arms_combine != U_Net.cat:
                assert all([e.out_channels == arm_channels for e in list_of_encoders])
            to_channels = 2 * in_channels if self.skip_combine == U_Net.cat else in_channels
            
            self.upsampling = ConvTranspose[self.dimension](in_channels * copies_of_inputs, in_channels, self.pooling_size, self.pooling_size, 0)
            block_params.update({'in_channels': to_channels})
            self.conv_block = Convolution_Block(**block_params)
            
            # arm_channels = list_of_encoders[0].out_channels
            # print(in_channels, [e.out_channels for e in list_of_encoders], self.skip_combine)
            # if self.skip_combine == U_Net.cat: to_channels = in_channels - arm_channels
            # else: assert all([in_channels == encoder.out_channels for encoder in list_of_encoders]); to_channels = in_channels
            # self.upsampling = ConvTranspose[self.dimension](in_channels * copies_of_inputs, to_channels, self.pooling_size, self.pooling_size, 0)
            # # block_params.update({'in_channels': to_channels + arm_channels, 'out_channels': out_channels}) # sum([encoder.out_channels for encoder in list_of_encoders])
            # self.conv_block = Convolution_Block(**block_params)

        def forward(self, x, enc_result):
            y = self.upsampling(x)
            if self.padding == self.kernel_size // 2:
                joint = combine([enc_result, bt.crop_as(y, enc_result)], self.skip_combine)
            else: joint = combine([bt.crop_as(enc_result, y), y], self.skip_combine)
            #     to_combine = list_of_encoder_results + [bt.crop_as(y, list_of_encoder_results[0])]
            # else: to_combine = [bt.crop_as(encoder_result, y) for encoder_result in list_of_encoder_results] + [y]
            # joint = combine(to_combine, self.skip_combine)
            return self.conv_block(joint)


    def __init__(self, **params):
        super().__init__()
        default_values = {'dimension': 2, 'depth': 4, 'conv_num': 2, 'padding': 'SAME', 
                          'in_channels': 1, 'out_channels': 2, 'block_channels': 64, 
                          'kernel_size': 3, 'pooling_size': 2, 'keep_prob': 0.5, 'conv_block': 'conv', 
                          'multi_arms': "shared", 'multi_arms_combine': U_Net.cat, 'skip_combine': U_Net.cat, 
                          'res_type': bt.add, 'activation': nn.ReLU, 'final_activation': Softmax, 
                          'initializer': "normal(0, 0.1)", 'cum_layers': -1, 
                          'bottleneck_in_channels': 0, 'bottleneck_out_channels': 0, 'variational': False}
        param_values = {}
        param_values.update(default_values)
        param_values.update(params)
        for k, v in param_values.items(): setattr(self, k, v)
        
        if isinstance(self.block_channels, int):
            self.block_channels = [self.in_channels] + [self.block_channels << min(i, 2 * self.depth - i) for i in range(2 * self.depth + 1)] + [self.out_channels]
        bchannels = self.block_channels
        if not callable(self.block_channels): self.block_channels = lambda i: bchannels[i + 1]
        
        if isinstance(self.padding, str): self.padding = {'SAME': self.kernel_size // 2, 'ZERO': 0, 'VALID': 0}.get(self.padding.upper(), self.kernel_size // 2)
        self.activation, self.final_activation = parse_activations(self.activation, self.final_activation, get_environ_vars())
        
        if isinstance(self.cum_layers, int): self.cum_layers = [self.cum_layers, self.cum_layers]
        l, u = self.cum_layers
        l = (l + 2 * self.depth + 1) % (2 * self.depth + 1)
        u = (u + 2 * self.depth + 1) % (2 * self.depth + 1)
        if l > u: l, u = u, l
        self.cum_layers = [max(l, self.depth), min(u, 2 * self.depth)]
        
        param_values = {k: getattr(self, k) for k in param_values if k not in ('in_channels', 'out_channels')}
        
        initialize_model, initialize_params = parse(self.initializer)
        self.arm_type, self.arm_num = parse(self.multi_arms)
        self.arm_num = 1 if len(self.arm_num) == 0 else self.arm_num[0]
        if self.arm_type == 'shared': self.dif_arm_num = 1
        else: self.dif_arm_num = self.arm_num
        
        for iarm in range(self.dif_arm_num):
            for k in range(self.depth + 1):
                setattr(self, 'block%d_%d' % (k, iarm), self.Encoder_Block(self.block_channels(k - 1), self.block_channels(k), k != 0, param_values))
        
        n_bottleneck_multiplier = 1
        if self.multi_arms_combine == U_Net.cat: n_bottleneck_multiplier = self.arm_num
        if self.bottleneck_in_channels > 0:
            setattr(self, 'bottleneck_in', Convolution_Block(self.bottleneck_in_channels, self.block_channels(self.depth), **param_values))
            if self.multi_arms_combine == U_Net.cat: n_bottleneck_multiplier += 1
        
        if self.bottleneck_out_channels > 0:
            setattr(self, 'bottleneck_out', Convolution_Block(self.block_channels(self.depth) * n_bottleneck_multiplier, self.bottleneck_out_channels, **param_values))

        for k in range(self.cum_layers[0], self.depth + 1):
            conv = Convolution[self.dimension](self.block_channels(k), self.block_channels(2 * self.depth + 1), 1, 1, 0)
            eval('nn.init.%s_' % initialize_model)(conv.weight, *initialize_params)
            if k < self.cum_layers[1]:
                setattr(self, 'block%dout' % k, nn.Sequential(conv, self.activation))
                setattr(self, 'out%dupsample' % k, ConvTranspose[self.dimension](
                    self.block_channels(2 * self.depth + 1), self.block_channels(2 * self.depth + 1), self.pooling_size, self.pooling_size, 0
                ))
            else: setattr(self, 'block%dout' % k, conv)

        for k in range(self.depth + 1, self.cum_layers[1] + 1):
            encoders = [getattr(self, 'block%d_%d' % (2 * self.depth - k, iarm)) for iarm in range(self.dif_arm_num)] * (self.arm_num // self.dif_arm_num)
            if k > self.depth + 1: n_bottleneck_multiplier = 1
            setattr(self, 'block%d' % k, self.Decoder_Block(
                encoders, 
                self.block_channels(k - 1), self.block_channels(k), 
                param_values, n_bottleneck_multiplier
            ))
            
            if self.multi_arms_combine == U_Net.cat:
                skip_in_channels = sum([e.out_channels for e in encoders])
            else: skip_in_channels = encoders[0].out_channels
            skip_out_channels = self.block_channels(k - 1)
            if self.variational: skip_out_channels *= 2
            conv = Convolution[self.dimension](skip_in_channels, skip_out_channels, self.kernel_size, self.kernel_size // 2, self.padding)
            eval('nn.init.%s_' % initialize_model)(conv.weight, *initialize_params)
            setattr(self, 'skipconv%d' % k, nn.Sequential(conv, self.activation))
            
            conv = Convolution[self.dimension](self.block_channels(k), self.block_channels(2 * self.depth + 1), 1, 1, 0)
            eval('nn.init.%s_' % initialize_model)(conv.weight, *initialize_params)
            if k < self.cum_layers[1]:
                setattr(self, 'block%dout' % k, nn.Sequential(conv, self.activation))
                setattr(self, 'out%dupsample' % k, ConvTranspose[self.dimension](
                    self.block_channels(2 * self.depth + 1), self.block_channels(2 * self.depth + 1), self.pooling_size, self.pooling_size, 0
                ))
            else: setattr(self, 'block%dout' % k, conv)
        
        self.to(bt.get_device().main_device)
        
    @property
    def bottleneck(self):
        if not getattr(self, 'new_bottleneck', False):
            raise RuntimeError("Attempting to get bottleneck from U_Net without any forward pass. ")
        self.new_bottleneck = False
        if self.bottleneck_out_channels > 0:
            result = getattr(self, f'block{self.depth}result')
            return self.bottleneck_out(result).mean(...)
        else: return

    def forward(self, x, bottleneck_input=None):
        size = x.size()[1:]
        if len(size) == self.dimension and self.in_channels == 1: x = x.unsqueeze([1])
        elif len(size) == self.dimension + 1 and self.in_channels * self.arm_num == size[0]: pass
        else: raise ValueError(f"The input tensor does not correspond to the U-Net structure: got {size}, but requires ([{self.in_channels * self.arm_num}], n_1, ⋯, n_{self.dimension}). ")
        
        assert size[0] % self.arm_num == 0
        inputs = x.split(size[0] // self.arm_num, 1)
        assert len(inputs) == self.arm_num
        
        for i, y in enumerate(inputs):
            for k in range(self.depth + 1):
                y = getattr(self, 'block%d_%d' % (k, 0 if self.arm_type == 'shared' else i))(y)
                setattr(self, 'block%d_%dresult' % (k, i), y)
        
        to_combine = [getattr(self, 'block%d_%dresult' % (self.depth, i)) for i in range(self.arm_num)]
        
        if self.bottleneck_in_channels > 0:
            assert bottleneck_input is not None
            to_combine.append(self.bottleneck_in(bottleneck_input))
        
        z = combine(to_combine, self.multi_arms_combine)
        setattr(self, 'block%dresult' % self.depth, z)
        self.new_bottleneck = True
        
        latents = []
        for k in range(self.depth + 1, self.cum_layers[1] + 1):
            to_skip = [getattr(self, 'block%d_%dresult' % (2 * self.depth - k, iarm)) for iarm in range(self.arm_num)]
            to_skip = [bt.crop_as(x, to_skip[0]) for x in to_skip]
            joint = combine(to_skip, self.multi_arms_combine)
            joint = getattr(self, 'skipconv%d' % k)(joint)
            latents.append(joint)
            if self.variational: joint = U_Net.variational_resample(joint)
            z = getattr(self, 'block%d' % k)(z, joint)
            setattr(self, 'block%dresult' % k, z)

        t = 0
        for k in range(self.cum_layers[0], self.cum_layers[1] + 1):
            setattr(self, 'block_out%dresult' % k, getattr(self, 'block%dout' % k)(getattr(self, 'block%dresult' % k)) + t)
            if k < self.cum_layers[1]: t = getattr(self, 'out%dupsample' % k)(getattr(self, 'block_out%dresult' % k))
        
        result = self.final_activation(getattr(self, 'block_out%dresult' % k))
        if self.variational: return result, latents
        return result
        
    def optimizer(self, lr=0.001): return bt.Optimization(bt.optim.Adam, self.parameters(), lr)

    def loss(self, x, y):
        y_hat = self(x)
        clamped = y_hat.clamp(1e-10, 1.0)
        self.y_hat = y_hat
        return - bt.sum(y * bt.log(clamped), 1).mean().mean()
        
    def __getitem__(self, i):
        if self.arm_num == 1 and i <= self.depth: i = (i, 0)
        return getattr(self, 'block%dresult' % i if isinstance(i, int) else 'block%d_%dresult' % i)
        
    def __iter__(self):
        for i in range(2 * self.depth + 1):
            if i <= self.depth:
                for iarm in range(self.arm_num):
                    yield 'block%d_%dresult' % (i, iarm), (i, iarm)
            else: yield 'block%dresult' % i, i

class CNN(U_Net):
    '''
    Args:
        dimension (int): The dimension of the images. Defaults to 2 (see VGG). 
        blocks (int): The number of the downsampling blocks. Defaults to 5 blocks (see VGG).
        conv_num (int or list of length 'blocks'): The number of continuous convolutions in one block. Defaults to [2, 2, 3, 3, 3] (see VGG).
            If the numbers for all blocks are the same, one can use one integer.
        padding (int or str): Indicate the type of padding used. Defaults to 'SAME' indicating a same output shape as the input. 
        in_channels (int): The number of channels for the input. Defaults to 1 (see VGG).
        out_elements (int): The number of channels for the output, as the number of classification. Defaults to 1000 for 1000 classes.
        layer_channels (int or list of length 'blocks'): The number of channels for each block. Defaults to [64, 128, 256, 512, 512] (see VGG). 
            Or else, a function may be provided to compute the output channels given the block index (-1 ~ 2 * depth + 1). 
        kernel_size (int): The size of the convolution kernels. Defaults to 3 (see VGG). 
        pooling_size (int): The size of the pooling kernels. Defaults to 2 (see VGG). 
        // keep_prob (float): The keep probability for the dropout layers. 
        conv_block (str): A string with possible values in ('conv', 'dense', 'residual'), indicating which kind of block the U-Net is using: normal convolution layers, DenseBlock or ResidualBlock. 
        multi_arms (str): A string with possible values in ('shared(2)', 'seperate(2)'), indicating which kind of encoder arms are used. 
        multi_arms_combine (function): The combining type for multi-arms. Defaults to catenation (cat). Other possible skip types include torch.mul or torch.add. 
        res_type (function): The combining type for the residual connections. It should be torch.add in most occasions. 
        activation (class or str): The activation function used after the convolution layers. Defaults to nn.ReLU. 
        final_activation (class or str): The activation function after the final convolution. 
        initializer (str): A string indicating the initialing strategy. Possible values are normal(0, 0.1) or uniform(-0.1, 0.1) or constant(0) (all parameters can be changed)
    '''
    
    def __init__(self, dimension = 2, blocks = 5, conv_num = 2, padding = 'SAME', 
        in_channels = 1, out_elements = 2, layer_channels = 64, kernel_size = 3, pooling_size = 2, 
        keep_prob = 0.5, conv_block = 'conv', multi_arms = "shared", multi_arms_combine = U_Net.cat, 
        res_type = bt.add, activation = nn.ReLU, final_activation = Softmax, initializer = "normal(0, 0.1)"):
        depth = blocks - 1
        activation, final_activation = parse_activations(activation, final_activation, get_environ_vars())
        if isinstance(layer_channels, int):
            maxlc = layer_channels
            layer_channels = [in_channels]
            multiplier = math.pow(maxlc / in_channels, 1 / (depth + 1))
            for i in range(depth):
                layer_channels.append(int(layer_channels[-1] * multiplier))
            layer_channels.append(maxlc)
            layer_channels.extend([0] * depth)
            layer_channels.append(out_elements)
        super().__init__(dimension = dimension, depth = depth, conv_num = conv_num, 
            padding = padding, in_channels = in_channels, out_channels = out_elements, 
            block_channels = layer_channels, kernel_size = kernel_size, 
            pooling_size = pooling_size, keep_prob = keep_prob, conv_block = conv_block,
            multi_arms = multi_arms, multi_arms_combine = multi_arms_combine, skip_combine = None,
            res_type = res_type, activation = activation, final_activation = final_activation,
            initializer = initializer, cum_layers = depth)

    def forward(self, x):
        fnact = self.final_activation
        self.final_activation = NoAct()
        r = fnact(super().forward(x).flatten(2).mean(-1))
        self.final_activation = fnact
        return r
        
class FCN(nn.Module):
    '''
    Fully connected network, with hidden layers of increased and then decreased sizes. 
        For layer_elements = 64 and layers = 8 and in_elements = out_elements = 8, 
        the layer sizes are [8, 16, 32, 64, 64, 32, 16, 8]. 
    
    Args:
        layers (int): Indicate the number of fully connected layers. 
        in_elements (int): The number of elements for the input. Defaults to 1.
        out_elements (int): The number of elements for the output, as the number of classification. Defaults to 1000 for 1000 classes.
        layer_elements (int or list of length 'layers'): The number of channels for each block. In a VGG, it should be [64, 128, 256, 512, 512]. 
            Or else, a function may be provided to compute the output channels given the block index (-1 ~ 2 * depth + 1). 
        kernel_size (int): The size of the convolution kernels. Defaults to 3. 
        keep_prob (float): The keep probability for the dropout layers. 
        activation (class or str): The activation function used after the convolution layers. Defaults to nn.ReLU. 
        final_activation (class or str): The activation function after the final convolution. 
        initializer (str): A string indicating the initialing strategy. Possible values are normal(0, 0.1) or uniform(-0.1, 0.1) or constant(0) (all parameters can be changed)
    '''

    def __init__(self, layers = 4, in_elements = 1, out_elements = 2, layer_elements = 64, 
        keep_prob = 0.5, activation = nn.ReLU, final_activation = ..., initializer = "normal(0, 0.1)"):
        activation, final_activation = parse_activations(activation, final_activation, get_environ_vars())
        if isinstance(layer_elements, int):
            maxlc = layer_elements
            layer_elements = [in_elements]
            multiplier = bt.pow(maxlc / in_elements, 1 / (layers // 2 - 1))
            for i in range(layers // 2 - 1):
                layer_elements.append(int(layer_elements[-1] * multiplier))
            layer_elements.append(maxlc)
            if layers % 2 == 0: layer_elements.extend(layer_elements[-2::-1])
            else: layer_elements.extend(layer_elements[::-1])
            layer_elements[-1] = out_elements
        if isinstance(layer_elements, list):
            lc = layer_elements.copy()
            layer_elements = lambda i: lc[i]
        self.layers = []
        for l in range(layers):
            fcl = nn.Linear(layer_elements(l), layer_elements(l+1))
            initialize_model, initialize_params = parse(initializer)
            eval('nn.init.%s_' % initialize_model)(fcl.weight, *initialize_params)
            self.layers.append(fcl)
            if l < layers - 1:
                self.layers.append(activation)
                self.layers.append(nn.Dropout(keep_prob))
            else: self.layers.append(final_activation)
        self.struct = nn.Sequential(*self.layers)
        self.to(bt.get_device().main_device)

    def forward(self, x):
        return self.struct(x)

class CoupledNeuralODEMarch(nn.Module):
    '''
    The default march function for Coupled Neural ODE. 
    
    Args:
        traj_channels (list of int):    The number of channels for the trajectories in ODE. Defaults to 2, indicating no coupled network.
        time_channels (int):            The number of channels for the time tensor. Defaults to 1.
        march_coupled (list bool):      Whether to march a coupled trajectory or not. Defaults to True. Being False is equivalent to input injection.
            Single boolean value controls all trajectories except the first. 
        shared_param_across_time (bool):
            Whether to share the parameters for different time point t. Defaults to True. 
            False is only available for auto_step_size = False and n_step is not None which indicates a fixed number of iterations. 
            When shared_param_across_time = False, we have the additional argument
            |   n_step (int):   The number of steps. Defaults to 50. 
    '''
    
    def __init__(self, traj_channels = 2, time_channels = 1, n_step = 50, 
                 march_coupled = True, shared_param_across_time = True, regularization = None, **params):
        super().__init__()
        
        if isinstance(traj_channels, int): traj_channels = [traj_channels]
        self.num_trajs = len(traj_channels)
        all_channels = sum(traj_channels)
        if isinstance(march_coupled, bool): march_coupled = [True] + [march_coupled] * (self.num_trajs - 1)
        
        self.traj_channels = traj_channels
        self.time_channels = time_channels
        self.march_coupled = march_coupled
        
        self.n_step = n_step
        self.shared = shared_param_across_time
        self.regularization = regularization
        self.reference = None
        params.update(dict(final_activation = None))
        
        for i_traj, traj_channel, march in zip(range(self.num_trajs), traj_channels, march_coupled):
            if march:
                if self.shared:
                    # Only one shared march function is needed for the recurrently accumulated neural network. 
                    setattr(self, f'march_func_traj{i_traj}', 
                            Convolution_Block(in_channels=all_channels + time_channels, out_channels=traj_channel, **params))
                else:
                    # `n_step` functions are defined for neural ODE with unshared parameters. 
                    for i in range(n_step):
                        setattr(self, f'march_func_traj{i_traj}_step{i}', 
                                Convolution_Block(in_channels=all_channels + time_channels, out_channels=traj_channel, **params))
        
        self.parameter_size = ByteSize(0)
        for p in self.parameters():
            self.parameter_size += bt.byte_size(p)
    
    def time(self, t):
        return t * bt.ones(self.reference.shape.with_n_channel(self.time_channels), dtype=self.reference.dtype, device=self.reference.device)
    
    class zero_grad(nn.Module):
        def __init__(self, x): super().__init__(); self.reference = x
        def forward(self, x): return 0 * self.reference
    
    def forward(self, i, t, *x):
        if i == 0: self.reference = x[0]
        self.memory_used = self.reference.byte_size()
        
        if not self.shared:
            assert i < self.n_step, f"Iteration {i} out of bound (≥n_step={self.n_step}) in the integration of Neural ODE. "
        time_tensor = self.time(t)
        inputs = bt.cat(*x, time_tensor, [])
        
        grads = []
        for i_traj, traj_x, march in zip(range(self.num_trajs), x, self.march_coupled):
            if march:
                if self.shared: grads.append(getattr(self, f"march_func_traj{i_traj}")(inputs))
                else: grads.append(getattr(self, f"march_func_traj{i_traj}_step{i}")(inputs))
            else: grads.append(bt.zeros_like(traj_x))
        self.memory_used += bt.get_memory_used(inputs, *grads)
        return bt.MultiVariate(*grads)
    
    def backward(self, i, t, *aug_x):
        assert self.reference is not None, "Cannot run backward before forward function is went through. "
        self.memory_used = ByteSize(0)
        symbol = lambda x: [u.detach().requires_grad_(True) for u in x]
        
        if not self.shared:
            assert i < self.n_step, f"Iteration {i} out of bound (≥n_step={self.n_step}) in the integration of Neural ODE. "
        time_tensor = self.time(t)
        
        trajs = aug_x[:self.num_trajs]
        adjs = aug_x[self.num_trajs:2*self.num_trajs]
        adj_params = aug_x[2*self.num_trajs:]
        self.memory_used += bt.get_memory_used(*trajs, *adjs, *adj_params)
        
        with bt.enable_grad():
            symbols = symbol(trajs)
            inputs = bt.cat(*symbols, time_tensor, [])
            funcs = []
            for i_traj, traj_x, march in zip(range(self.num_trajs), trajs, self.march_coupled):
                if march:
                    if self.shared: funcs.append(getattr(self, f"march_func_traj{i_traj}"))
                    else: funcs.append(getattr(self, f"march_func_traj{i_traj}_step{i}"))
                else: funcs.append(CoupledNeuralODEMarch.zero_grad(traj_x))
            grads = [f(inputs) for f in funcs]
            symbol_grads = symbol(grads)
            self.memory_used += bt.get_memory_used(*symbols, *grads, *symbol_grads)
            
            if self.regularization is None: reg_loss = None
            else: reg_loss = self.regularization(i, t, *symbols, *symbol_grads)
            reg_adjs = bt.grad(reg_loss, (*symbols, *symbol_grads), None, allow_unused = True, retain_graph = True)
            reg_adjs_x = reg_adjs[:self.num_trajs]
            reg_adjs_grad = reg_adjs[self.num_trajs:]
            self.memory_used += bt.get_memory_used(reg_loss, *reg_adjs)
            
            grad_adjs = bt.grad(grads, symbols + sum([list(f.parameters()) for f in funcs], []), [u+v for u, v in zip(adjs, reg_adjs_grad)], allow_unused = True, retain_graph = True)
            grad_adjs_x = grad_adjs[:self.num_trajs]
            grad_adjs_params = grad_adjs[self.num_trajs:]
            self.memory_used += bt.get_memory_used(*grad_adjs)
            
        return bt.MultiVariate(*grads).detach(), -bt.MultiVariate(*[u+v for u, v in zip(grad_adjs_x, reg_adjs_x)], *grad_adjs_params)

class NeuralODEMarch(nn.Module):
    '''
    The default march function for Neural ODE. 
    
    Args:
        main_channels (int):    The number of channels for the main stream ODE. Defaults to 2.
        coupled_channels (int): The number of channels for the coupled stream ODE. Defaults to 0, indicating no coupled network. coupled_channels = True is equivalent to coupled_channels = 64. 
        time_channels (int):    The number of channels for the time tensor. Defaults to 1.
        march_coupled (bool):   Whether to march the coupled trajectory or not. Defaults to True. Being False is equivalent to input injection.
        shared_param_across_time (bool):
            Whether to share the parameters for different time point t. Defaults to True. 
            False is only available for auto_step_size = False and n_step is not None which indicates a fixed number of iterations. 
            When shared_param_across_time = False, we have the additional argument
            |   n_step (int):   The number of steps. Defaults to 50. 
    '''
    
    def __init__(self, main_channels = 2, coupled_channels = 0, time_channels = 1, n_step = 50, 
                 march_coupled = True, shared_param_across_time = True, regularization = None, **params):
        super().__init__()
        
        self.main_channels = main_channels
        self.coupled_channels = coupled_channels
        self.time_channels = time_channels
        self.n_step = n_step
        self.march_coupled = march_coupled
        self.shared = shared_param_across_time
        self.regularization = regularization
        
        self.has_coupled_func = coupled_channels > 0 and march_coupled
        self.reference = None
        params.update(dict(final_activation = None))
        
        if self.shared:
            # Only one shared march function is needed for the recurrently accumulated neural network. 
            self.main_march_func = Convolution_Block(in_channels=main_channels + coupled_channels + time_channels, out_channels=main_channels, **params)
            if self.has_coupled_func:
                self.coupled_march_func = Convolution_Block(in_channels=main_channels + coupled_channels + time_channels, out_channels=coupled_channels, **params)
        else:
            # `n_step` functions are defined for neural ODE with unshared parameters. 
            for i in range(n_step):
                setattr(self, f'main_march_func_{i}', Convolution_Block(in_channels=main_channels + coupled_channels + time_channels, out_channels=main_channels, **params))
                if self.has_coupled_func:
                    setattr(self, f'coupled_march_func_{i}', Convolution_Block(in_channels=main_channels + coupled_channels + time_channels, out_channels=coupled_channels, **params))
        
        self.parameter_size = ByteSize(0)
        for p in self.parameters():
            self.parameter_size += bt.byte_size(p)
    
    def time(self, t):
        return t * bt.ones(self.reference.shape.with_n_channel(self.time_channels), dtype=self.reference.dtype, device=self.reference.device)
    
    def forward(self, i, t, *x):
        if i == 0: self.reference = x[0]
        self.memory_used = self.reference.byte_size()
        
        if self.shared:
            main_march_func = self.main_march_func
            if self.has_coupled_func:
                coupled_march_func = self.coupled_march_func
        else:
            assert i < self.n_step, f"Iteration {i} out of bound (≥n_step={self.n_step}) in the integration of Neural ODE. "
            main_march_func = getattr(self, f'main_march_func_{i}')
            if self.has_coupled_func:
                coupled_march_func = getattr(self, f'coupled_march_func_{i}')
        
        time_tensor = self.time(t)
        if self.coupled_channels == 0: model = "NODE"
        elif self.march_coupled: model = "CNODE"
        else: model = "SNODE"
        
        inputs = bt.cat(*x, time_tensor, [])
        # print(inputs)
        # inputs = bt.cat(*(u*u*u for u in x), time_tensor, [])
        if model == "NODE":
            grad_main = main_march_func(inputs)
            self.memory_used += bt.get_memory_used(inputs, grad_main)
            return bt.MultiVariate(grad_main)
        elif model == "SNODE":
            grad_main = main_march_func(inputs)
            grad_coupled = bt.zeros_like(traj_coupled)
            self.memory_used += bt.get_memory_used(inputs, grad_main, grad_coupled)
            return bt.MultiVariate(grad_main, grad_coupled)
        elif model == "CNODE":
            grad_main = main_march_func(inputs)
            grad_coupled = coupled_march_func(inputs)
            self.memory_used += bt.get_memory_used(inputs, grad_main, grad_coupled)
            return bt.MultiVariate(grad_main, grad_coupled)
        else: raise RuntimeError(f"Unrecognized model {model}. ")
    
    def backward(self, i, t, *aug_x):
        assert self.reference is not None, "Cannot run backward before forward function is went through. "
        self.memory_used = ByteSize(0)
        
        if self.shared:
            main_march_func = self.main_march_func
            if self.has_coupled_func:
                coupled_march_func = self.coupled_march_func
        else:
            assert i < self.n_step, f"Iteration {i} out of bound (≥n_step={self.n_step}) in the integration of Neural ODE. "
            main_march_func = getattr(self, f'main_march_func_{i}')
            if self.has_coupled_func:
                coupled_march_func = getattr(self, f'coupled_march_func_{i}')
        
        time_tensor = self.time(t)
        if self.coupled_channels == 0: model = "NODE"
        elif self.march_coupled: model = "CNODE"
        else: model = "SNODE"
        
        symbol = lambda u: u.detach().requires_grad_(True)
        if model == "NODE":
            traj_main, adj_main, *adj_params = aug_x
            self.memory_used += bt.get_memory_used(traj_main, adj_main, *adj_params)
            
            with bt.enable_grad():
                symbol_main = symbol(traj_main)
                grad_main = main_march_func(bt.cat(symbol_main, time_tensor, []))
                symbol_grad_main = symbol(grad_main)
                self.memory_used += bt.get_memory_used(symbol_main, grad_main, symbol_grad_main)
                
                if self.regularization is None: reg_loss = None
                else: reg_loss = self.regularization(i, t, symbol_main, symbol_grad_main)
                reg_main, reg_grad_main = bt.grad(reg_loss, (symbol_main, symbol_grad_main), None, allow_unused = True, retain_graph = True)
                self.memory_used += bt.get_memory_used(reg_loss, reg_main, reg_grad_main)
                
                grad_adj_main, *grad_adj_params = bt.grad(grad_main, (symbol_main, *main_march_func.parameters()), adj_main + reg_grad_main, allow_unused = True, retain_graph = True)
                self.memory_used += bt.get_memory_used(grad_adj_main, *grad_adj_params)
            
            return bt.MultiVariate(grad_main).detach(), bt.MultiVariate(grad_adj_main + reg_main, *grad_adj_params)

        elif model == "SNODE":
            traj_main, traj_coupled, adj_main, adj_coupled, *adj_params = aug_x
            self.memory_used += bt.get_memory_used(traj_main, traj_coupled, adj_main, adj_coupled, *adj_params)
            with bt.enable_grad():
                symbol_main, symbol_coupled = symbol(traj_main), symbol(traj_coupled)
                grad_main = main_march_func(bt.cat(symbol_main, symbol_coupled, time_tensor, []))
                symbol_grad_main = symbol(grad_main)
                self.memory_used += bt.get_memory_used(symbol_main, symbol_coupled, grad_main, symbol_grad_main)
                
                if self.regularization is None: reg_loss = None
                else: reg_loss = self.regularization(i, t, symbol_main, symbol_coupled, symbol_grad_main)
                reg_main, reg_coupled, reg_grad_main = bt.grad(reg_loss, (symbol_main, symbol_coupled, symbol_grad_main), None, allow_unused = True, retain_graph = True)
                self.memory_used += bt.get_memory_used(reg_loss, reg_main, reg_coupled, reg_grad_main)
                
                grad_adj_main, grad_adj_coupled, *grad_adj_params = bt.grad(grad_main, (symbol_main, symbol_coupled, *main_march_func.parameters()), adj_main + reg_grad_main, allow_unused = True, retain_graph = True)
                self.memory_used += bt.get_memory_used(grad_adj_main, grad_adj_coupled, *grad_adj_params)
            
            return bt.MultiVariate(grad_main.detach(), bt.zeros_like(traj_coupled)), -bt.MultiVariate(grad_adj_main + reg_main, grad_adj_coupled + reg_coupled, *grad_adj_params)
        
        elif model == "CNODE":
            traj_main, traj_coupled, adj_main, adj_coupled, *adj_params = aug_x
            self.memory_used += bt.get_memory_used(traj_main, traj_coupled, adj_main, adj_coupled, *adj_params)
            
            with bt.enable_grad():
                symbol_main, symbol_coupled = symbol(traj_main), symbol(traj_coupled)
                grad_main = main_march_func(bt.cat(symbol_main, symbol_coupled, time_tensor, []))
                grad_coupled = coupled_march_func(bt.cat(symbol_main, symbol_coupled, time_tensor, []))
                symbol_grad_main = symbol(grad_main)
                symbol_grad_coupled = symbol(grad_coupled)
                self.memory_used += bt.get_memory_used(symbol_main, symbol_coupled, grad_main, grad_coupled, symbol_grad_main, symbol_grad_coupled)
                
                if self.regularization is None: reg_loss = None
                else: reg_loss = self.regularization(i, t, symbol_main, symbol_coupled, symbol_grad_main, symbol_grad_coupled)
                reg_main, reg_coupled, reg_grad_main, reg_grad_coupled = bt.grad(reg_loss, (symbol_main, symbol_coupled, symbol_grad_main, symbol_grad_coupled), None, allow_unused = True, retain_graph = True)
                self.memory_used += bt.get_memory_used(reg_loss, reg_main, reg_coupled, reg_grad_main, reg_grad_coupled)
                
                grad_adj_main, grad_adj_coupled, *grad_adj_params = bt.grad(
                    (grad_main, grad_coupled), 
                    (symbol_main, symbol_coupled, *main_march_func.parameters(), *coupled_march_func.parameters()), 
                    (adj_main + reg_grad_main, adj_coupled + reg_grad_coupled), 
                    allow_unused = True, retain_graph = True)
                self.memory_used += bt.get_memory_used(grad_adj_main, grad_adj_coupled, *grad_adj_params)
            
            return bt.MultiVariate(grad_main, grad_coupled).detach(), -bt.MultiVariate(grad_adj_main + reg_main, grad_adj_coupled + reg_coupled, *grad_adj_params)
        
        else: raise RuntimeError(f"Unrecognized model {model}. ")

class NeuralODE(nn.Module):
    '''
    Neural ODE structue. 
    Please use MultiVariate for coupled ode trajectories. 
    
    Use the following table to create frameworks of corresponding structures.
    NODE: unsupervised ODE network, the standard ODE framework.                 coupled_channels = 0
    SNODE: supervised ODE, by inserting the features to the ODE framework.      coupled_channels > 0; coupled_ODE = False
    CNODE: coupled ODE, supervise the ODE framework by a coupled ODE sequence.  coupled_channels > 0; coupled_ODE = True
    RCNODE: NODE versions with regularization.                                  regularization != None
    
    Args:
    
    Systematic Args:
        
        march_func (nn.Module):
            The gradient estimator for the ODE. 
            ⚠️ NOTE that the Module should have:
                A method `forward(i, t, x) -> dx` for forward propagation;
                A method `backward(i, t, adjx) -> dadjx` for the adjoint backward propagation. 
            If not specified, the following arguments are available and it defaults to a simple stack of `conv_num` convolution layers. 
            |   dimension (int):                The dimension of the images. 
            |   main_channels (int):            The number of channels for the main stream ODE. Defaults to 2.
            |   coupled_channels (int):         The number of channels for the coupled stream ODE. Defaults to 0, indicating no coupled network. coupled_channels = True is equivalent to coupled_channels = 64. 
            |   mid_channels (int):             The number of channels for the hidden layers in main stream. Defaults to `main_channels`.
            |   time_channels (int):            The number of channels for the time tensor. Defaults to 1.
            |   march_coupled (bool):           Whether to march the coupled trajectory or not. Defaults to True. Being False is equivalent to input injection.
            |   shared_param_across_time (bool):
            |       Whether to share the parameters for different time point t. Defaults to True. 
            |       False is only available for auto_step_size = False and n_step is not None which indicates a fixed number of iterations. 
            |
            |   conv_num (int):                 The number of continuous convolutions in each layer. Defaults to 2. 
            |   kernel_size (int):              The size of the convolution kernels. Defaults to 3. 
            |   padding (int or str):           Indicate the type of padding used. Defaults to 'SAME' indicating a same output shape as the input. 
            |   conv_block (str):               A string with possible values in ('conv', 'dense', 'residual'), indicating which kind of block for the layers: normal convolution layers, DenseBlock or ResidualBlock. 
            |   res_type (function):            The combining type for the residual connections. It should be torch.add in most occasions. 
            |   linear_layer (bool):            Whether the convolutions are used for 1x1 images (equivalent to linear layers). 
            |   activation (class):             The activation function used after the convolution layers. Defaults to nn.ReLU. 
            |   final_activation (class):       The activation function after the final convolution. Defaults to self.activation. 
            |   initializer (str):              A string indicating the initialing strategy. Possible values are normal(0, 0.1) or uniform(-0.1, 0.1) or constant(0) (all parameters can be changed)
        
        (Debug)
        simultaneously_print (bool):            Whether to verbosely print the loggings. Defaults to False. 
    
    Setting Args:
        method (str):                           The gradient estimation method. Defaults to 'Euler'. 
            A string with possible values in ('Euler', 'RK2', 'RK4'). 
        start_time (float):                     The starting time. Defaults to 0. 
        end_time (float):                       The ending time. Defaults to None. 
        step_through (str):                     The criterion to obtain the ouput value at time t when marching through it. Defaults to 'through'. 
            A string with possible values in ('through', 'cut', 'stop'). 
            |   'through':                      Go through time t and use an interpolation for the resulting value at t; 
            |   'cut':                          March by a gradient estimated by the current step size, but use the interpolation as the resulting value at t; 
            |   'stop':                         Make time t as a time point so that the step should stop here to perform accurate estimation. 
            Note that, 'cut' and 'stop' are equivalent when method = 'Euler'. 
        interpolation (str):                    The interpolation method for step_through. Defaults to 'Bezier'. 
            A string with possible values in ('linear', 'spline', 'Bezier', 'SymBezier'). 
            |   'linear': [with f'(t₀)]         f(t) = (1-s)*f(t₀) + s*f(t₁), s=(t-t₀)/(t₁-t₀); 
            |   'spline':                       f(t) = [t³ t² t 1]a, where a = A⁻¹v, v = [f(t₀) f(t₁) f'(t₀) f'(t₁)]ᵀ,
            |     [with both f'(t₀) and f'(t₁)]     A = [[t₀³ t₀² t₀ 1], [t₁³ t₁² t₁ 1], [3t₀² 2t₀ 1 0], [3t₁² 2t₁ 1 0]]
            |   'Bezier':                       f(t) = (1-s)²*f(t₀) + 2s(1-s)*m + s²*f(t₁), s=(t-t₀)/(t₁-t₀), 
            |     [with both f'(t₀) and f'(t₁)]     m = 1/2 * [f(t₀) + f(t₁) + λf'(t₀) + μf'(t₁)], 
            |                                       λ = <βf'(t₀) - γf'(t₁), f(t₁) - f(t₀)> / (αβ-γ²), 
            |                                       μ = <γf'(t₀) - αf'(t₁), f(t₁) - f(t₀)> / (αβ-γ²), 
            |                                       α = ‖f'(t₀)‖², β = ‖f'(t₁)‖², γ = <f'(t₀), f'(t₁)>; 
            |                                       Note that λ should be > 0 and μ < 0 to ensure the monotonous of the interpolation function. 
            |                                       > For other cases, we use f(t) = (1-s)³*f(t₀) + 3s(1-s)²*m + 3s²(1-s)*n + s³*f(t₁), s=(t-t₀)/(t₁-t₀) instead, 
            |                                       > where m = f(t₀) + (λ - √β/α μ)/2 * f'(t₀) and n = f(t₁) + (μ - √α/β λ)/2 * f'(t₁), 
            |   'SymBezier': [with f'(t₀)]      f(t) = (1-s)²*f(t₀) + 2s(1-s)*m + s²*f(t₁), s=(t-t₀)/(t₁-t₀), 
            |                                       m = f(t₀) + 1/2 * λf'(t₀), 
            |                                       λ = ‖f(t₁) - f(t₀)‖² / <f'(t₀), f(t₁) - f(t₀)>. 
            Note that the interpolation methods using both gradients spend more time. 
        auto_step_size (bool):
            Whether to auto decide the time step. Defaults to True. 
            When auto_step_size = True, we have additional arguments
            |   step_size (float):              The initial marching step size. Defaults to 0.02. 
            |   min_step (float):               The minimal step size in auto adaptive steps. Defaults to 0.
            |   max_step (float):               The maximal step size in auto adaptive steps. Defaults to 1.
            |   safety (float):                 The safety measure to avoid step size explosion. Defaults to 0.9. 
            |   order (float):                  The order of error ratio. Defaults to 0.9. 
            |   min_factor (float):             The minimal factor value. Defaults to 0.2.
            |   max_factor (float):             The maximal factor value. Defaults to 10.
            |   max_n_step (int):               The maximal number of steps. Defaults to 1000. 
            |   adjust_later_if_accurate        Whether to adjust the step size in the succeeding iteration if the current accuracy is enough. 
            |                     (bool):       It will slightly improve the speed but degrade the accuracy. Defaults to False. 
            > rtol and atol indicate a tolerance of `rtol x magnitute + atol` error for the adjustment of auto step size. 
            |   * rtol (float):                 The relative tolerance for auto layer and stopping criteria. 
            |   * atol (float):                 The absolute tolerance for auto layer and stopping criteria. 
            When auto_step_size = False, we have additional arguments
            |   step_size (float):              The marching step size, 0 means the direct Euler steps among gaps in forward agument `ts`. Defaults to 0.02. 
            |   n_step (int):                   The number of steps. Defaults to None. (It should ensure that start_time + n_step * step_size ≤ end_time.)
        
        error_ratio_tolerance (float):          The tolerant value for error ratio which indicates an overflow of estimation error. It is commonly > 1. Defaults to 100.
        regularization (func):                  The regularization term for the delta terms: regularization(i, t, dt, x, dx) -> r;
                                                for coupled_channels > 0 and not march_coupled, use the arguments (i, t, dt, x_main, x_coupled, dx_main).
                                                for coupled_channels > 0 and march_coupled, use the arguments (i, t, dt, x_main, x_coupled, dx_main, dx_coupled).
        use_adjoint (bool):                     Whether to backpropagate through adjoint method (True) or pytorch autograd system (False). Defaults to True. 
        detached (bool):                        Whether the model is a constant model that does not require training or not. Defaults to False. 
    '''
    
    class Interpolator:
        def __init__(self, t0, x0, grad0: "can be None", t1, x1, grad1: "can be None", method="Bezier"):
            self.method = method
            self.t0 = t0
            self.t1 = t1
            if grad0 is not None:
                grad0 = grad0 * bt.abs(t1 - t0)
                main_dim = [dim for dim in [bt.FeatDim, bt.SpaceDim, bt.SequeDim] if grad0.has_dim(dim)][0]
            if grad1 is not None: grad1 = grad1 * bt.abs(t1 - t0)
            
            if self.method == "linear":
                self.func = lambda s: x0 * (1-s) + x1 * s
            elif self.method == "spline":
                assert grad0 is not None and grad1 is not None, "Cubic spline interpolator for NeuralODE.Interpolator requires both grads to be present. "
                norm = lambda x: bt.sqrt((x * x).sum(main_dim, keepdim=True))
                dx = x1 - x0
                grad0 = norm(dx) * grad0 / norm(grad0)
                grad1 = norm(dx) * grad1 / norm(grad1)
                a3 = grad0 + grad1 - 2 * dx
                a2 = 3 * dx - 2 * grad0 - grad1
                a1 = grad0
                a0 = x0
                self.func = lambda s: a3 * s*s*s + a2 * s*s + a1 * s + a0
            elif self.method == "Bezier":
                assert grad0 is not None, "Bezier interpolator for NeuralODE.Interpolator requires grad0 to be present. "
                iprod = lambda x, y: (x * y).sum(main_dim, keepdim=True)
                if grad1 is None:
                    l = iprod(x1-x0, x1-x0) / iprod(grad0, x1-x0)
                    m = x0 + l * grad0 / 2
                    self.func = lambda s: x0 * (1-s)*(1-s) + m * 2*s*(1-s) + x1 * s*s
                else:
                    a = iprod(grad0, grad0)
                    b = iprod(grad1, grad1)
                    r = iprod(grad0, grad1)
                    lm = bt.where(bt.equals(a*b, r*r), bt.ones_like(grad0), iprod(grad0 * b - grad1 * r, x1-x0) / (a*b-r*r))
                    mu = bt.where(bt.equals(a*b, r*r), -bt.ones_like(grad0), iprod(grad0 * r - grad1 * a, x1-x0) / (a*b-r*r))
                    # middle point for 3-point Bezier
                    m = (x0 + x1 + lm * grad0 + mu * grad1) / 2
                    # 1st middle point for 4-point Bezier
                    u = x0 + (lm - mu * bt.sqrt(b/a)) * grad0 / 2
                    # 2nd middle point for 4-point Bezier
                    v = x1 + (mu - lm * bt.sqrt(a/b)) * grad1 / 2
                    self.func = lambda s: bt.where(
                        (lm > 0) & (mu < 0), 
                        x0 * (1-s)*(1-s) + m * 2*s*(1-s) + x1 * s*s, 
                        x0 * (1-s)*(1-s)*(1-s) + u * 3*s*(1-s)*(1-s) + v * 3*s*s*(1-s) + x1 * s*s*s
                    )
            else: raise TypeError(f"Unrecognized method '{self.method}' for NeuralODE.Interpolator. ")
            
        def __call__(self, t):
            s = (t - self.t0) / (self.t1 - self.t0)
            if not isinstance(s, bt.Tensor): s = bt.tensor(s)
            return self.func(s)
    
    class Stepper:
        class Euler:
            order = 1
            butcher_table = [[0     , 0     ], 
                             [0     , 1     ]]
        class Heun: # alias name: RK2
            order = 2
            butcher_table = [[0     , 0     , 0     ], 
                             [1     , 1     , 0     ], 
                             [0     , 1 / 2 , 1 / 2 ]]
        class SSP: # alias name: RK3
            order = 3
            butcher_table = [[0     , 0     , 0     , 0     ], 
                             [1     , 1     , 0     , 0     ], 
                             [1 / 2 , 1 / 4 , 1 / 4 , 0     ], 
                             [0     , 1 / 6 , 1 / 6 , 2 / 3 ]]
        class RK4:
            order = 4
            butcher_table = [[0     , 0     , 0     , 0     , 0     ], 
                             [1 / 2 , 1 / 2 , 0     , 0     , 0     ], 
                             [1 / 2 , 0     , 1 / 2 , 0     , 0     ], 
                             [1     , 0     , 0     , 1     , 0     ], 
                             [0     , 1 / 6 , 1 / 3 , 1 / 3 , 1 / 6 ]]
        class matlabODE45: # alias name: RK5
            order = 5
            butcher_table = [[0     , 0         , 0         , 0         , 0             , 0         , 0     ], 
                             [1 / 4 , 1 / 4     , 0         , 0         , 0             , 0         , 0     ], 
                             [3 / 8 , 3 / 32    , 9 / 32    , 0         , 0             , 0         , 0     ], 
                             [12/13 , 1932/2197 , -7200/2197, 7296/2197 , 0             , 0         , 0     ], 
                             [1     , 439 / 216 , -8        , 3680/513  , -845 / 4104   , 0         , 0     ], 
                             [1 / 2 , -8 / 27   , 2         , -3544/2565, 1859 / 4104   , -11 / 40  , 0     ], 
                             [0     , 16 / 135  , 0         , 6656/12825, 28561 / 56430 , -9 / 50   , 2 / 55]]
        
        def __init__(self, func):
            self.memory_used = ByteSize(0)
            self.func = func
        
        def __call__(self, i, t, dt, x: bt.MultiVariate, method: str):
            if isinstance(method, str): method = getattr(self, method)
            self.memory_used = ByteSize(0)
            num_eq = len(method.butcher_table) - 1
            Ks = []
            for i in range(num_eq + 1):
                ci, *ai_ = method.butcher_table[i]
                if len(Ks) == 0: grad = bt.zeros_like(x)
                else: grad = bt.tensor_to(bt.stack([a * k for a, k in zip(ai_, Ks)], bt.FuncDim).sum(bt.FuncDim), x)
                if i == num_eq: break
                Ks.append(self.get_grad(i, t + ci * dt, x + dt * grad))
            return bt.MultiVariate(grad, inherited=x.inherited)

        def get_grad(self, i, t, x):
            grad = self.func(i, t, *x)
            self.memory_used += getattr(self.func, 'memory_used', 0)
            if isinstance(grad, tuple): return bt.MultiVariate(*grad)
            return bt.MultiVariate(grad)
    
    def __init__(self,
            # Systematic Args
            march_func = None, 
                main_channels = 2, 
                coupled_channels = 0, 
                mid_channels = None, 
                time_channels = 1, 
                
                march_coupled = True, 
                shared_param_across_time = True, 
                
                dimension = 2, conv_num = 3, kernel_size = 3, padding = 1, conv_block = 'conv', res_type = bt.add, linear_layer = False, 
                activation = nn.ReLU, final_activation = ..., initializer = "normal(0, 0.1)", 
                
            simultaneously_print = False,
            
            # Setting Args
            method = 'Euler', 
            start_time = 0., 
            end_time = None, 
            step_through = 'through', 
            interpolation = 'Bezier', 
            auto_step_size = True, 
                step_size = 0.02, 
                
                min_step = 0.,      max_step = 1., 
                safety = 0.9,       order = 0.9, 
                min_factor=0.2,     max_factor=10., 
                max_n_step = 1000, 
                rtol = 1e-3,        atol = 1e-3, 
                adjust_later_if_accurate = False,
                
                n_step = None,
                
            error_ratio_tolerance = 1e2, 
            regularization = None, 
            use_adjoint = True, 
            detached = False
        ): # NeuralODE.__init__
        super().__init__()
        
        if coupled_channels == True: coupled_channels = 64
        if mid_channels is None: mid_channels = main_channels
        
        conv_params = dict(dimension=dimension, mid_channels=mid_channels, conv_num=conv_num, 
            kernel_size=kernel_size, padding=padding, conv_block=conv_block, res_type=res_type, linear_layer=linear_layer, 
            activation=activation, final_activation=final_activation, initializer=initializer)
        
        if not shared_param_across_time:
            if auto_step_size:          raise TypeError(f"NeuralODE must share parameters across time when the number of steps are not fixed (currently auto_step_size={auto_step_size}). ")
            elif n_step is not None:    raise TypeError(f"NeuralODE must share parameters across time when the number of steps are not fixed (currently n_step={n_step}). ")
        
        if n_step is not None:
            step_end_time = start_time + n_step * step_size
            if end_time is None: end_time = step_end_time
            elif end_time < step_end_time: raise TypeError(f"NeuralODE taking an end time (={end_time}) before the end of steps ({step_end_time}). ")
        
        if march_func is None: march_func = NeuralODEMarch(
            main_channels = main_channels, coupled_channels = coupled_channels, time_channels = time_channels, n_step = n_step, 
            march_coupled = march_coupled, shared_param_across_time = shared_param_across_time, **conv_params)
        
        self.march_func = march_func
        self.simultaneously_print = simultaneously_print
        self.error_ratio_tolerance = error_ratio_tolerance
        self.regularization = regularization
        self.use_adjoint = use_adjoint
        self.detached = detached
        
        step_method = dict(RK1='Euler', RK2='Heun', RK3='SSP', RK5='matlabODE45').get(method, method)
        setting_args = dict(step_method = step_method, 
                            start_time = start_time, end_time = end_time, 
                            step_through = step_through, interpolation = interpolation, 
                            auto_step_size = auto_step_size, step_size = step_size)
        if setting_args['auto_step_size']:
            setting_args.update(dict(min_step = min_step, max_step = max_step, safety = safety, order = order, min_factor = min_factor, max_factor = max_factor, 
                max_n_step = max_n_step, rtol = rtol, atol = atol, adjust_later_if_accurate = adjust_later_if_accurate))
        else: setting_args['n_step'] = n_step
        
        self.setting_args = setting_args
    
    def time(self, t, n_channel):
        return t * bt.ones(self.reference.shape.with_feature((n_channel,)), dtype=self.reference.dtype, device=self.reference.device)
    
    def error_ratio(self, x0, x1, norm=None):
        if norm is None: norm = lambda x: x.abs().max()
        error = norm(x1 - x0)
        tolerance = self.setting_args['rtol'] * bt.max(norm(x0), norm(x1)) + self.setting_args['atol']
        return (error / tolerance).mean()
    
    def integrate(self,
        diff_func,                      # The marching function that estimates the difference. 
        x0: bt.MultiVariate,            # The initial value. 
        t_in: bt.Tensor = None,         # (float or Tensor) The start time points: the time points that start from and accepts gradient input. Note that it is required to be ordered. 
        t_out: bt.Tensor = None,        # (float or Tensor) The estimated end time points, which will be stacked to the output. Note that it is required to be ordered. 
        collect_grads: bool = False,    # Whether to collect the gradients from "model.sequence_grads". Defaults to False. 
        save_trajectory: bool = False   # Whether to save trajectory or not. Defaults to False. 
    ): # NeuralODE.integrate
        """ Integrate from t_in[0] to each time point in t_out. """
        stime = lambda t: f"•{t}•"
        self.memory_used = ByteSize(0)
        self.reference = x0[0]
        self.trajectory = []
        
        stepper = NeuralODE.Stepper(diff_func)
        
        if t_in is None:
            t_in = self.setting_args['start_time']
        if t_out is None:
            assert self.setting_args['end_time'] is not None, "`end_time` is required for NeuralODE integration without specific `t_out`. "
            t_out = self.setting_args['end_time']
        if not isinstance(t_in, bt.Tensor): t_in = bt.tensor(t_in)
        if not isinstance(t_out, bt.Tensor): t_out = bt.tensor(t_out)
        t_in = t_in.view(-1); t_out = t_out.view(-1)
        
        i_iter = 0
        current_time = t_in[0]
        if self.setting_args['step_size'] == 0:
            if self.setting_args['auto_step_size']:
                raise TypeError("`step_size` cannot be 0, unless auto_step_size=False when it represents Euler steps. ")
            euler_step = True
        else: euler_step = False
        current_step_size = self.setting_args['step_size']
        
        self.log = SPrint(print_onscreen=self.simultaneously_print)
        if t_in[0] < t_out[-1]:
            int_dir = bt.tensor(+1.)
            self.log(f"[Integrating a forward process...]")
        else:
            int_dir = bt.tensor(-1.)
            self.log(f"[Integrating an inversed process...]")
        
        later = lambda t, target: t > target if int_dir > 0 else t < target
        current_step_size = int_dir * current_step_size
        
        ts = bt.cat(t_in, t_out)
        assert bt.all(int_dir * (ts[1:]-ts[:-1]) >= -bt.eps), f"The time points for `NeuralODE.integrate` should be monotonous yet {ts} is not. "
        output_time_index = t_in.size(0)
        upcoming_time_index = 0
        upcoming_time = ts[upcoming_time_index]
        
        current_value = x0
        output_values = []
        
        while True:
            
            if not euler_step and bt.equals(current_step_size, 0):
                raise RuntimeError(f"[Step size underflow] Step size is {stime(current_step_size)} at iteration [{i_iter}] and time {stime(current_time)}. ")
            
            if int_dir < 0 and current_time < self.setting_args['start_time'] or \
                self.setting_args['end_time'] is not None and int_dir > 0 and current_time > self.setting_args['end_time']:
                raise RuntimeError(f"[Stopping the iterations] Time {stime(current_time)} out of range [{stime(self.setting_args['start_time'])} ~ {stime(self.setting_args['end_time'])}]. ")
            
            # Record the current step. 
            if int_dir > 0: i = i_iter
            else: i = int(self.n_iter - i_iter - 1)
            self.log(f"[{i_iter:^4d}] Iteration at time [{i}]{stime(current_time)}({'+' if int_dir > 0 else ''}{current_step_size}), targeting {stime(upcoming_time)}. ")
            self.log(f"     └ * with value{'s' if len(current_value) > 1 else ''} {current_value.__raw_repr__()}. ".replace('\n', ''))
            
            # Adjust the step size. 
            grad_evaluated = False
            if self.setting_args['auto_step_size']:
                # Try and adjust the step size
                step_method = self.setting_args['step_method']
                ref_method = dict(Heun='SSP').get(step_method, "Heun") # The reference method for tolerance estimation
                # March forward
                step_grad = stepper(i, current_time, current_step_size, current_value, step_method)
                ref_grad = stepper(i, current_time, current_step_size, current_value, ref_method)
                error_ratio = self.error_ratio(step_grad, ref_grad) # error / tolerance
                if self.setting_args['adjust_later_if_accurate'] and error_ratio < 1:
                    self.log(f"     | Gradient accepted at an error ratio = {error_ratio} given grad({step_method})={step_grad} and reference({ref_method})={ref_grad}")
                    grad_evaluated = True
                
                factor = self.setting_args['safety'] * error_ratio ** (-self.setting_args['order'])
                factor = min(max(factor, self.setting_args['min_factor']), self.setting_args['max_factor']) # clamp the factor, within [0.2, 10] by default. 
                next_step_size = bt.sign(current_step_size) * (current_step_size * factor).abs().clamp(self.setting_args['min_step'], self.setting_args['max_step'])
                if bt.equals(next_step_size, self.setting_args['min_step']) and error_ratio > self.error_ratio_tolerance:
                    raise RuntimeError(f"[Step size overflow] Error ratio limit exceeded ({error_ratio} > {self.error_ratio_tolerance}) for step {current_step_size} at iteration {i_iter} and time [{i}]{stime(current_time)}. Reduce step size for more accurate estimations. ")
                self.log(f"     └ Adjusting step size {current_step_size} >> {next_step_size} by error ratio = {error_ratio}.")
                self.log(f"     • step grad ({step_method})={step_grad.__raw_repr__()}") # tokenize(str(step_grad).split('(', 1)[-1], sep=',')[0]
                self.log(f"     • reference grad ({ref_method})={ref_grad.__raw_repr__()}. ")
                current_step_size = next_step_size
            elif euler_step: current_step_size = upcoming_time - current_time
            
            # Record the time points that should be decided in the very semi-interval. 
            to_record = [] # The time points in [current_time, next_time) that requires output
            cut_at = None
            complete = False
            while bt.equals(upcoming_time, current_time):
                to_record.append((upcoming_time_index, current_time))
                upcoming_time_index += 1
                if upcoming_time_index < len(ts):
                    upcoming_time = ts[upcoming_time_index]
                else: complete = True; break
            while later(current_time + current_step_size, upcoming_time):
                if complete: break
                if not self.setting_args['auto_step_size']:
                    assert self.setting_args['step_through'] == "stop", "`step_through` should be 'stop' unless `auto_step_size` is activated. "
                if self.setting_args['step_through'] == "through":
                    to_record.append((upcoming_time_index, upcoming_time))
                    upcoming_time_index += 1
                    if upcoming_time_index >= len(ts): complete = True; break
                    upcoming_time = ts[upcoming_time_index]
                else:
                    if self.setting_args['step_through'] == "cut": cut_at = upcoming_time
                    elif self.setting_args['step_through'] == "stop": current_step_size = upcoming_time - current_time; grad_evaluated = False
                    else: raise TypeError(f"Unrecoginized argument step_through='{self.setting_args['step_through']}'. ")
                    break
            
            for i, t in to_record:
                if i < output_time_index: # Collect gradients when 
                    if collect_grads and getattr(self, 'sequence_grads', None) is not None:
                        num_trajs = current_value.inherited['num_trajs']
                        current_value[num_trajs:2*num_trajs] = current_value[num_trajs:2*num_trajs] + self.sequence_grads.pick(output_time_index - i - 1, '')
                else: break
            
            # Gradient evaluation
            if grad_evaluated: current_grad = step_grad
            else: current_grad = stepper(i, current_time, current_step_size, current_value, self.setting_args['step_method'])
            self.memory_used += stepper.memory_used
            
            # Create conditions in format (t, x, grad)
            # retrieve cons0 and update
            conds0 = (current_time, current_value, current_grad)
            next_time = current_time + current_step_size
            next_value = current_value + current_step_size * current_grad
            
            inptor = None
            for i, t in to_record:
                if i >= output_time_index: # Record outputs 
                    if later(t, current_time):
                        if int_dir < 0:
                            raise TypeError(f"Cannot send gradient bact at time points when empirical interpolation method is performed during step {self.setting_args['step_through']}s, use step_through = cut/stop instead. ")
                        if inptor is None:
                            # retrieve conds1
                            if self.setting_args['interpolation'] in ('spline', 'Bezier') and len(to_record) > 0:
                                # requires the gradient at end time. 
                                next_grad = stepper.get_grad(i, next_time, next_value)
                                conds1 = (next_time, next_value, next_grad)
                            else: conds1 = (next_time, next_value, None)
                            inptor = NeuralODE.Interpolator(*conds0, *conds1, method=self.setting_args['interpolation'])
                        mid_value = inptor(t)
                        output_values.append(mid_value)
                        self.log(f"[{'*':^4s}] Recording trajectory value at time {stime(t)} by an interpolation between {stime(current_time)} and {stime(next_time)}. ")
                        self.log(f"     └ * with value{'s' if len(mid_value) > 1 else ''} {mid_value.__raw_repr__()}. ")
                    else:
                        self.log(f"[{'*':^4s}] Recording trajectory value at time {stime(t)}(≈{stime(current_time)}). ")
                        output_values.append(current_value)
            
            if save_trajectory:
                self.trajectory.append(current_value)
            
            if cut_at is not None:
                current_value = current_value + (cut_at - current_time) * current_grad
                # current_step_size remains
                current_time = cut_at
            else:
                current_value = next_value
                current_time = next_time
            
            if complete:
                self.log(f"[Stopping the iterations] Last required time point reached. ")
                break
            if self.setting_args['auto_step_size']:
                max_n_step = self.setting_args['max_n_step']
                if i_iter >= max_n_step:
                    raise RuntimeError(f"[Stopping the iterations] Maximal steps ({max_n_step}) reached. ")
            else:
                n_step = self.setting_args['n_step']
                if n_step is not None and i_iter >= n_step:
                    self.log(f"[Stopping the iterations] Scheduled steps ({n_step}) reached. ")
                    break
            i_iter += 1
        
        if save_trajectory:
            self.trajectory.append(current_value)
        
        self.n_iter = i_iter
        if len(output_values) != len(t_out): raise RuntimeError(f"Unmatched output values (of length {len(output_values)}) w.r.t. required time points (of length {len(t_out)}). ")
        if int_dir > 0: self.output = output_values[-1]
        if len(output_values) == 1: return output_values[0]
        return bt.stack(output_values, '')
    
    def forward_func(self, init_x, init_time=None, output_time=None, save_trajectory=False):
        self.init_time = init_time
        self.output_time = output_time
        if save_trajectory is None: save_trajectory = getattr(self, 'save_trajectory', False)
        self.save_trajectory = save_trajectory
        return self.integrate(self.march_func.forward, init_x, init_time, output_time, save_trajectory=save_trajectory)
    
    def backward_func(self, grads, save_trajectory=None):
        if self.detached: raise RuntimeError("Cannot perform backpropagation for detached model. ")
        if grads.ndim > self.output.ndim:
            self.sequence_grads = grads.as_subclass(bt.Tensor).special_from(self.output.unsqueeze(''))
        else: self.sequence_grads = grads.as_subclass(bt.Tensor).special_from(self.output).unsqueeze('')
        output_grad = bt.MultiVariate(bt.zeros_like(self.output), inherited=self.output.inherited)
        
        aug_x = bt.MultiVariate(self.output.detach(), output_grad, bt.MultiVariate(bt.zeros_like(p) for p in self.march_func.parameters()))
        num_trajs = len(self.output)
        aug_x.inherited['num_trajs'] = num_trajs
        if save_trajectory is None: save_trajectory = getattr(self, 'save_BPtrajectory', False)
        self.save_BPtrajectory = save_trajectory
        if isinstance(self.output_time, (int, float)): output_time = self.output_time
        else: output_time = self.output_time.flip()
        res = self.integrate(self.march_func.backward, aug_x, output_time, self.init_time, collect_grads=True, save_trajectory=save_trajectory)
        for p, g in zip(self.march_func.parameters(), res[2*num_trajs:]): p.grad = g.mean({})
        
        self.sequence_grads = None
        return res[num_trajs:2*num_trajs]
    
    class Autograd_Func(bt.autograd.Function):
        @staticmethod
        def forward(ctx, self, init_x, init_time, output_time):
            ctx.self = self
            with bt.no_grad():
                return self.forward_func(init_x, init_time, output_time, save_trajectory=self.save_trajectory)
        
        @staticmethod
        def backward(ctx, sequence_grads):
            in_grad = out_grad = None
            # TODO: the gradients for input/output_time is required for latent ODE. 
            return None, ctx.self.backward_func(sequence_grads), in_grad, out_grad
    
    def detach(self): self.detached = True; return self
    
    def forward(self, *init_x, init_time=None, output_time=None, requires_sequence=False, save_trajectory=False, save_BPtrajectory=False):
        # Pack init_x
        init_x = bt.MultiVariate(*init_x)
        
        # Default init/output time
        if init_time is None: init_time = self.setting_args['start_time']
        if output_time is None:
            end_time = self.setting_args['end_time']
            step = self.setting_args['step_size']
            freq = math.ceil(1. / step)
            if end_time is None: end_time = self.setting_args['start_time'] + freq * step
            if not requires_sequence: output_time = end_time
            else: output_time = bt.arange(freq + 1) * step + self.setting_args['start_time']
        
        if not self.use_adjoint:
            ret = self.forward_func(init_x, init_time, output_time, save_trajectory=save_trajectory)
        else:
            if not self.detached:
                # Make init_x requires gradient to ensure torch go through the backward function
                x_has_grad = init_x.requires_grad
                if not x_has_grad: init_x.requires_grad = True
            
            self.save_trajectory = save_trajectory
            self.save_BPtrajectory = save_BPtrajectory
            ret = self.Autograd_Func.apply(self, init_x, init_time, output_time)
            
            if not self.detached and not x_has_grad:
                # recover the state of init_x. 
                init_x.grad = None
                init_x.requires_grad = False
        
        return bt.MultiVariate(ret, inherited=init_x.inherited).with_sequence_size(ret.n_time if ret.has_time else None)

class RNN(nn.Module):
    '''
    Recurrent Neural Network structue. 
    
    Args:
        dimension (int): The dimension of the images. 
        layers (int): The number recurrent layers. 
        conv_num (int): The number of continuous convolutions in each layer. Defaults to 2. 
        main_channels (int): The number of channels for the main sequence. Defaults to 2.
        hidden_channels (int): The number of channels for the hidden features. Defaults to 0, indicating no coupled network. 
        mid_channels (int): The number of channels for the hidden layers. Defaults to 10.
        kernel_size (int): The size of the convolution kernels. Defaults to 3. 
        padding (int or str): Indicate the type of padding used. Defaults to 'SAME' indicating a same output shape as the input. 
        conv_block (str): A string with possible values in ('conv', 'dense', 'residual'), indicating which kind of block for the layers: normal convolution layers, DenseBlock or ResidualBlock. 
        res_type (function): The combining type for the residual connections. It should be torch.add in most occasions. 
        activation (class): The activation function used after the convolution layers. Defaults to nn.ReLU. 
        final_activation (class): The activation function after the final convolution. Defaults to self.activation. 
        initializer (str): A string indicating the initialing strategy. Possible values are normal(0, 0.1) or uniform(-0.1, 0.1) or constant(0) (all parameters can be changed)
    '''
    
    def __init__(self, dimension = 2, layers = 10, conv_num = 2, kernel_size = 3, padding = 1, frequency = 50, linear_layer = False,
        main_channels = 2, hidden_channels = 0, mid_channels = None, conv_block = 'conv', res_type = bt.add, 
        activation = nn.ReLU, final_activation = ..., initializer = "normal(0, 0.1)", regularization = None):
        super().__init__()
        
        if hidden_channels == True: hidden_channels = 64
        if mid_channels is None: mid_channels = main_channels
        
        params = dict(dimension=dimension, mid_channels=mid_channels, linear_layer=linear_layer, conv_num=conv_num, kernel_size=kernel_size, padding=padding, 
            activation=activation, final_activation=final_activation, conv_block=conv_block, res_type=res_type, initializer=initializer)
        avouch(layers >= 0, "RNN network should have fixed number of layers. ")
        if hidden_channels > 0: self.init_hidden = None
        self.step_layer = Convolution_Block(main_channels + hidden_channels, main_channels + hidden_channels, **params)
        
        self.n_layer = layers
        self.hidden_channels = hidden_channels
    
    def forward(self, input_x, init_hidden=None, n_output_step=None):
        """input_x: ({n_batch}, 'steps', [n_feature], n_1, n_2)"""
        self.memory_used = ByteSize(0)
        n_batch, n_input_step, n_feature, *shape = input_x.shape
        if self.hidden_channels > 0:
            self.init_hidden = bt.zeros({n_batch}, [self.hidden_channels], *shape)
            self.memory_used += self.init_hidden.byte_size()
        if n_output_step is None: n_output_step = n_input_step
        hidden = self.init_hidden
        output_x = None
        outputs = []
        for i in range(self.n_layer):
            if i < n_input_step: layer_input = input_x[:, i]
            else: layer_input = output_x
            output = self.step_layer(bt.cat(layer_input, hidden, []))
            self.memory_used += getattr(self.step_layer, 'memory_used', 0)
            output_x = output[:, :n_feature]
            hidden = output[:, n_feature:]
            if i >= self.n_layer - n_output_step: outputs.append(output_x)
        return bt.stack(outputs, ''), hidden

class Models(bt.nn.Module):
    
    def __init__(self, **sub_models):
        super().__init__()
        self.sub_models = sub_models
        self.device = sub_models.pop('device', bt.default_device())
        for name, model in sub_models.items():
            setattr(self, name, getattr(model, 'to', lambda d: model)(self.device))
        self.best_score = 0
    
    def initialize(self, weight, bias=None):
        if bias is None: bias = weight
        def initializer(layer):
            if hasattr(layer, 'weight') and isinstance(layer.weight, bt.torch.Tensor):
                initialize_model, initialize_params = parse(weight)
                getattr(layer.weight.data, f'{initialize_model}_')(*initialize_params)
            if hasattr(layer, 'bias') and isinstance(layer.bias, bt.torch.Tensor):
                initialize_model, initialize_params = parse(bias)
                getattr(layer.bias.data, f'{initialize_model}_')(*initialize_params)
        for name in self.sub_models:
            if not name or name.startswith('#'): continue
            getattr(getattr(self, name), 'apply', lambda i: ...)(initializer)
            
    def load(self, directory, epoch='best', **kwargs):
        checkpoint_dir = Path(directory)
        if isinstance(epoch, str):
            candidates = [ckpt_file for ckpt_file in checkpoint_dir if ckpt_file.name.endswith(epoch)]
            if len(candidates) == 0: raise FileNotFoundError(f"Cannot find checkpoint with label {epoch}. ")
            if len(candidates) > 1: raise FileNotFoundError(f"Multiple checkpoints found with label {epoch}: {candidates}. ")
            ckpt_file = candidates[0]
            token = ckpt_file.name.replace('epoch', '').replace(epoch, '').strip().strip('_')
        else: ckpt_file = f"epoch{epoch}.ckpt"
        kwargs.setdefault('weights_only', True)
        self.load_state_dict(bt.load(checkpoint_dir / ckpt_file, map_location=self.device, **kwargs))
        return touch(lambda: eval(token), None)
            
    def save(self, directory, epoch, score=0):
        avouch(0 <= score <= 1, TypeError("Model score should be between [0, 1] in method 'save'. "))
        checkpoint_dir = Path(directory).mkdir()
        if score > self.best_score:
            for ckpt_file in checkpoint_dir:
                if ckpt_file.name.endswith('_best'): ckpt_file.remove()
            bt.save(self.state_dict(), checkpoint_dir / f"epoch{epoch}_best.ckpt")
            self.best_score = score
        bt.save(self.state_dict(), checkpoint_dir / f"epoch{epoch}.ckpt")

if __name__ == "__main__":
#    unet = U_Net(multi_arms="seperate(3)", block_channels=16)
#    print(unet(bt.rand(10, 3, 100, 100)).size())
#    print(*[x + ' ' + str(unet[i].size()) for x, i in unet], sep='\n')
    unet = U_Net(
        dimension=3, 
        in_channels=2, 
        out_channels=3, 
        block_channels=4, 
        final_activation=None, 
        initializer="normal(0.0, 0.9)", 
#        conv_block='dense', 
#        conv_num=4, 
    )
    print(unet(bt.rand(10, 2, 50, 50, 50)).size())
    print(*[x + ' ' + str(unet[i].size()) for x, i in unet], sep='\n')