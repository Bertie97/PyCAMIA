
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
""".split()
    
import math

with __info__:
    import batorch as bt
    from batorch import nn

def parse(string):
    if string.count('(') > 1 or string.count(')') > 1: raise TypeError("Invalid to parse: " + string + ". ")
    if string.count('(') == 0 and string.count(')') == 0: string += '()'
    return eval('("' + string.lower().replace('(', '", (').replace(')', ',)').replace('(,)', '()') + ')')

def cat(*tensors): return bt.cat(tensors, 1)
def combine(list_of_items, reduction):
    if len(list_of_items) >= 2:
        z = reduction(list_of_items[0], list_of_items[1])
        for i in range(2, len(list_of_items)):
            z = reduction(z, list_of_items[i])
    else: z = list_of_items[0]
    return z

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
        default_values = {'dimension': 2, 'conv_num': 1, 'padding': 1, 'kernel_size': 3, 'conv_block': 'conv', 'res_type': bt.add, 'activation_function': nn.ReLU, 'active_args': {}}
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

    def forward(self, x):
        if self.conv_block == 'dense':
            conv_results = [x]
            conv_layer = True
            for layer in self.layers:
                if conv_layer: x = layer(bt.cat([bt.crop_as(l, conv_results[-1]) for l in conv_results], 1))
                else: x = layer(x)
                conv_layer = layer.__class__.__name__.startswith('Conv')
                if conv_layer: conv_results.append(x)
            return self.activation_function(**self.active_args)(x)
        else:
            y = x
            for layer in self.layers: y = layer(y)
            if self.conv_block == 'residual': z = self.res_type(bt.crop_as(x, y), y)
            else: z = y
            return self.activation_function(**self.active_args)(z)
        

class U_Net(nn.Module):
    
    class Softmax(nn.Module):
        def forward(self, x): return nn.functional.softmax(x, 1)
    
    class Encoder_Block(nn.Module):
        
        def __init__(self, in_channels, out_channels, has_pooling, params):
            super().__init__()
            block_params = params.copy()
            block_params.update({'in_channels': in_channels, 'out_channels': out_channels, 'has_pooling': has_pooling})
            self.__dict__.update(block_params)
            if has_pooling: self.pooling = eval('nn.MaxPool%dd' % self.dimension)(self.pooling_size, ceil_mode = True)
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
            self.__dict__.update(block_params)
            if self.skip_type == cat: to_channels = in_channels - list_of_encoders[0].out_channels
            else: assert all([in_channels == encoder.out_channels for encoder in list_of_encoders]); to_channels = in_channels
            self.upsampling = eval('nn.ConvTranspose%dd' % self.dimension)(in_channels * copies_of_inputs, to_channels, self.pooling_size, self.pooling_size, 0)
            block_params.update({'in_channels': to_channels + sum([encoder.out_channels for encoder in list_of_encoders]), 'out_channels': out_channels})
            self.conv_block = Convolution_Block(**block_params)

        def forward(self, x, list_of_encoder_results):
            y = self.upsampling(x)
            if self.padding == self.kernel_size // 2:
                to_combine = list_of_encoder_results + [bt.crop_as(y, list_of_encoder_results[0])]
            else: to_combine = [bt.crop_as(encoder_result, y) for encoder_result in list_of_encoder_results] + [y]
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
            block_channels (int): The number of channels for the first block if a number is provided. In a conventional U-Net, it should be 64. 
                If a list is provided, the length should be the same as the number of blocks plus two (2 * depth + 3). It represents the channels before and after each block (with the output channels included). 
                Or else, a function may be provided to compute the output channels given the block index (-1 ~ 2 * depth + 1) [including input_channels at -1 and output_channels at 2 * depth + 1]. 
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
            cum_layers (list): A list consisting two numbers [n, m] indicating that the result would be a summation of the upsamples of the results of the nth to the mth (included) blocks, block_numbers are in range 0 ~ 2 * depth. 
                The negative indices are allowed to indicate the blocks in a inversed order with -1 representing the output for the last block. 
        '''
        super().__init__()
        default_values = {'dimension': 2, 'depth': 4, 'conv_num': 2, 'padding': 'SAME', 'in_channels': 1, 'out_channels': 2, 'block_channels': 64, 'kernel_size': 3, 'pooling_size': 2, 'keep_prob': 0.5, 'conv_block': 'conv', 'multi_arms': "shared", 'multi_arms_combine': cat, 'skip_type': cat, 'res_type': bt.add, 'activation_function': nn.ReLU, 'active_args': {}, 'initializer': "normal(0, 0.1)", 'with_softmax': True, 'cum_layers': -1}
        param_values = {}
        param_values.update(default_values)
        param_values.update(params)
        self.__dict__.update(param_values)
        
        if isinstance(self.block_channels, int):
            self.block_channels = [self.in_channels] + [self.block_channels << min(i, 2 * self.depth - i) for i in range(2 * self.depth + 1)] + [self.out_channels]
        bchannels = self.block_channels
        if not callable(self.block_channels): self.block_channels = lambda i: bchannels[i + 1]
        
        if isinstance(self.padding, str): self.padding = {'SAME': self.kernel_size // 2, 'ZERO': 0, 'VALID': 0}.get(self.padding.upper(), self.kernel_size // 2)
        
        if isinstance(self.cum_layers, int): self.cum_layers = [self.cum_layers, self.cum_layers]
        l, u = self.cum_layers
        l = (l + 2 * self.depth + 1) % (2 * self.depth + 1)
        u = (u + 2 * self.depth + 1) % (2 * self.depth + 1)
        if l > u: l, u = u, l
        self.cum_layers = [max(l, self.depth), min(u, 2 * self.depth)]
        
        param_values = {k: self.__dict__[k] for k in param_values}
        
        self.arm_type, self.arm_num = parse(self.multi_arms)
        self.arm_num = 1 if len(self.arm_num) == 0 else self.arm_num[0]
        if self.arm_type == 'shared': self.dif_arm_num = 1
        else: self.dif_arm_num = self.arm_num
        
        for iarm in range(self.dif_arm_num):
            for k in range(self.depth + 1):
                setattr(self, 'block%d_%d' % (k, iarm), self.Encoder_Block(self.block_channels(k - 1), self.block_channels(k), k != 0, param_values))

        for k in range(self.cum_layers[0], self.depth + 1):
            conv = eval('nn.Conv%dd' % self.dimension)(self.block_channels(k), self.block_channels(2 * self.depth + 1), 1, 1, 0)
            initialize_model, initialize_params = parse(self.initializer)
            eval('nn.init.%s_' % initialize_model)(conv.weight, *initialize_params)
            if k < self.cum_layers[1]:
                setattr(self, 'block%dout' % k, nn.Sequential(conv, self.activation_function(**self.active_args)))
                setattr(self, 'out%dupsample' % k, eval('nn.ConvTranspose%dd' % self.dimension)(
                    self.block_channels(2 * self.depth + 1), self.block_channels(2 * self.depth + 1), self.pooling_size, self.pooling_size, 0
                ))
            else: setattr(self, 'block%dout' % k, conv)

        for k in range(self.depth + 1, self.cum_layers[1] + 1):
            setattr(self, 'block%d' % k, self.Decoder_Block(
                [getattr(self, 'block%d_%d' % (2 * self.depth - k, iarm)) for iarm in range(self.dif_arm_num)] * (self.arm_num // self.dif_arm_num), 
                self.block_channels(k - 1), self.block_channels(k), param_values, 
                self.arm_num if k == self.depth + 1 and self.multi_arms_combine == cat else 1
            ))
            conv = eval('nn.Conv%dd' % self.dimension)(self.block_channels(k), self.block_channels(2 * self.depth + 1), 1, 1, 0)
            initialize_model, initialize_params = parse(self.initializer)
            eval('nn.init.%s_' % initialize_model)(conv.weight, *initialize_params)
            if k < self.cum_layers[1]:
                setattr(self, 'block%dout' % k, nn.Sequential(conv, self.activation_function(**self.active_args)))
                setattr(self, 'out%dupsample' % k, eval('nn.ConvTranspose%dd' % self.dimension)(
                    self.block_channels(2 * self.depth + 1), self.block_channels(2 * self.depth + 1), self.pooling_size, self.pooling_size, 0
                ))
            else: setattr(self, 'block%dout' % k, conv)

        if self.with_softmax: self.softmax = self.Softmax()

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
        setattr(self, 'block%dresult' % self.depth, z)
            
        for k in range(self.depth + 1, self.cum_layers[1] + 1):
            z = getattr(self, 'block%d' % k)(z, [getattr(self, 'block%d_%dresult' % (2 * self.depth - k, iarm)) for iarm in range(self.arm_num)])
            setattr(self, 'block%dresult' % k, z)
            
        t = 0
        for k in range(self.cum_layers[0], self.cum_layers[1] + 1):
            setattr(self, 'block_out%dresult' % k, getattr(self, 'block%dout' % k)(getattr(self, 'block%dresult' % k)) + t)
            if k < self.cum_layers[1]: t = getattr(self, 'out%dupsample' % k)(getattr(self, 'block_out%dresult' % k))
        
        if self.with_softmax: return self.softmax(getattr(self, 'block_out%dresult' % k))
        else: return getattr(self, 'block_out%dresult' % k)
        
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
    def __init__(self, dimension = 2, blocks = 5, conv_num = 2, padding = 'SAME', 
        in_channels = 1, out_elements = 2, layer_channels = 64, kernel_size = 3, 
        pooling_size = 2, keep_prob = 0.5, conv_block = 'conv', multi_arms = "shared", 
        multi_arms_combine = cat, res_type = bt.add, activation_function = nn.ReLU,
        active_args = {}, initializer = "normal(0, 0.1)", with_softmax = True):
        '''
        ::paramerters:
            dimension (int): The dimension of the images. For conventional VGG, it is 2. 
            blocks (int): The number of the downsampling blocks. The conventional VGG has 5 blocks.
            conv_num (int or list of length 'blocks'): The number of continuous convolutions in one block. In VGG, this is [2, 2, 3, 3, 3]. If the numbers for all blocks are the same, one can use one integer.
            padding (int or str): Indicate the type of padding used. In a conventional VGG, padding is 'SAME' indicating . 
            in_channels (int): The number of channels for the input. In a conventional VGG, it should be 1.
            out_elements (int): The number of channels for the output, as the number of classification. In a conventional VGG, it should be 1000 for 1000 classes.
            layer_channels (int or list of length 'blocks'): The number of channels for each block. In a VGG, it should be [64, 128, 256, 512, 512]. 
                Or else, a function may be provided to compute the output channels given the block index (-1 ~ 2 * depth + 1). 
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
            cum_layers (list): A list consisting two numbers [n, m] indicating that the result would be a summation of the upsamples of the results of the nth to the mth (included) blocks, block_numbers are in range 0 ~ 2 * depth. The negative indices are allowed to indicate the blocks in a inversed order with -1 representing the output for the last block. 
        '''
        depth = blocks - 1
        if isinstance(layer_channels, int):
            maxlc = layer_channels
            layer_channels = [in_channels]
            multiplier = int(math.pow(maxlc / in_channels, 1 / (depth + 1)))
            for i in range(depth):
                layer_channels.append(layer_channels[-1] * multiplier)
            layer_channels.append(maxlc)
            layer_channels.extend([0] * depth)
            layer_channels.append(out_elements)
        super().__init__(dimension = dimension, depth = depth, conv_num = conv_num, 
            padding = padding, in_channels = in_channels, out_channels = out_elements, 
            block_channels = layer_channels, kernel_size = kernel_size, 
            pooling_size = pooling_size, keep_prob = keep_prob, conv_block = conv_block,
            multi_arms = multi_arms, multi_arms_combine = multi_arms_combine, skip_type = None,
            res_type = res_type, activation_function = activation_function, active_args = active_args,
            initializer = initializer, with_softmax = with_softmax, cum_layers = depth)

    def forward(self, x):
        wsm = self.with_softmax
        self.with_softmax = False
        if wsm: r = self.softmax(super().forward(x).flatten(2).mean(-1))
        else: r = super().forward(x).flatten(2).mean(-1)
        self.with_softmax = wsm
        return r
        
class FCN(nn.Module):

    class Softmax(nn.Module):
        def forward(self, x): return nn.functional.softmax(x, 1)
    def __init__(self, layers = 4, in_elements = 1, out_elements = 2, layer_elements = 64, 
        keep_prob = 0.5, activation_function = nn.ReLU, active_args = {}, 
        initializer = "normal(0, 0.1)", with_softmax = True):
        if isinstance(layer_elements, int):
            maxlc = layer_elements
            layer_elements = [in_elements]
            multiplier = int(bt.pow(maxlc / in_elements, 1 / (layers // 2 + 1)))
            for i in range(layers // 2 - 1):
                layer_elements.append(layer_elements[-1] * multiplier)
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
                self.layers.append(activation_function(**active_args))
                self.layers.append(nn.Dropout(keep_prob))
            elif with_softmax:
                self.layers.append(self.Softmax())
        self.struct = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.struct(x)


if __name__ == "__main__":
#    unet = U_Net(multi_arms="seperate(3)", block_channels=16)
#    print(unet(bt.rand(10, 3, 100, 100)).size())
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
    print(unet(bt.rand(10, 2, 50, 50, 50)).size())
    print(*[x + ' ' + str(unet[i].size()) for x, i in unet], sep='\n')