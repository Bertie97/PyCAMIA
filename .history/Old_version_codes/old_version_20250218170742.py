
############################################################
## This is the dump yard for outdated codes. 
## The script may not able to run bug-freely.
############################################################

## micomputing.interpolation: micomputing/trans.py:~1527 -[2023-08-21]
@alias("resample")
def interpolation(
        image: bt.Tensor, 
        trans: Callable = None, 
        method: str = 'Linear', 
        target_space: tuple = None,
        fill: (str, int, float) = 0,
        derivative: bool = False
    ):
    '''
    Interpolate using backward transformation.
    i.e. Compute the image I s.t. trans(x) = y for x in I and y in input image using interpolation method:
        method = Linear: Bilinear interpolation
        method = Nearest [NO GRADIENT!!!]: Nearest interpolation

    Params:
        image [bt.Tensor]: The target image.
            size: ([n_batch:optional], {n_channel:optional}, m@1, m@2, ..., m@n_dim)
        trans [Function or micomputing.SpatialTransformation]: Transformation function, mapping
            size: ([n_batch:optional], {n_dim}, n@1, n@2, ..., n@n_dim) to ([n_batch], {n_dim}, n@1, n@2, ..., n@n_dim)
        method [str: linear|nearest]: The interpolation method. 
        target_space [tuple or bt.Tensor]:
            Size (tuple) of a target ROI at the center of image. 
            OR Transformed coordinate space (bt.Tensor) of the output image. 
            size: length(n_dim) or ([n_batch:optional], {n_dim:optional}, size@1, size@2, ..., size@r)
        fill [str: nearest|background or int or float(number)]: Indicate the way to fill background outside `Surrounding`. 
        derivative [bool]: Whether to return the gradient. One can omit it when using torch.autograd.

        output [bt.Tensor]: The transformed image.
            size: ([n_batch], {n_channel:optional}, m@1, m@2, ..., m@n_dim)
            or when `target_space` is defined by tensor. 
            size: ([n_batch], size@1, size@2, ..., size@n_dim)
            or the derivative for the interpolation. (if `derivative = True`)
            size: ([n_batch], {n_dim}, size@1, size@2, ..., size@n_dim)

    Examples:
    ----------
    >>> Image = bt.rand(3, 100, 120, 80)
    >>> AM = bt.rand(4, 4)
    >>> AM[3, :] = bt.one_hot(-1, 4)
    >>> interpolation(Image, Affine(AM), method='Linear')
    '''
    # Deal with input `trans` and special dimensions of `image`
    if trans is not None and not trans.backward:
        if hasattr(trans, 'inv'): trans = trans.inv().fake_inv()
        else:
            print("Warning: Forward transformation found in method `interpolation`. Using `interpolation_forward` instead. ")
            return interpolation_forward(image, trans, method = method, target_space = target_space, fill = fill)
    image = bt.to_device(batorch_tensor(image))
    shape_out = image.shape
    if trans is None or trans.n_dim is None:
        if not image.has_batch:
            image = image.unsqueeze([])
        if not image.has_channel:
            image = image.standard().unsqueeze({1})
        n_dim = image.n_space # Get the spatial rank.
    else:
        n_dim = trans.n_dim
        if image.n_dim == n_dim:
            if image.has_special:
                print(f"Warning: 'interpolation' trying to transform {image.n_space}+{image.n_special}D image (with batch or channel) by {n_dim}D transformation, auto-removing special dimensions.")
                image.remove_special_()
            image = image.unsqueeze([]).unsqueeze_({})
        elif image.n_dim == n_dim + 1:
            if not image.has_batch:
                if image.has_channel: image = image.unsqueeze([])
                else: image = image.with_batch_dim(0).unsqueeze({1})
            elif not image.has_channel:
                image = image.unsqueeze({})
            else:
                print(f"Warning: 'interpolation' trying to transform {image.n_space}+{image.n_special}D image (with batch or channel) by {n_dim}D transformation, auto-removing the channel dimensions.")
                image = image.with_channeldim(None).unsqueeze({})
        elif image.n_dim == n_dim + 2:
            # _channal/batch dimensions used here as they are n_dim when not existed. 
            if image.n_special == 1:
                print(f"Warning: 'interpolation' trying to transform {image.n_space}+1D image (with batch or channel) by {n_dim}D transformation, auto-inserting new special dimension.")
            if not image.has_batch: image.batch_dimension = 0 if image._channel_dimension > 0 else 1
            if not image.has_channel: image.channel_dimension = 0 if image._batch_dimension > 0 else 1
    avouch(image.has_batch and image.has_channel, "Please use batorch tensor of size " +
            "([n_batch], {n_channel/n_feature:optional}, m_1, m_2, ..., m_r) [r=n_dim] for " + 
            f"data to be spatially interpolated, instead of {image.shape}. ")
    if trans is not None:
        avouch(image.n_batch == 1 or trans.n_batch in (None, image.n_batch, 1), "Please use transformation of a " +
            f"suitable n_batch to transform image with batchsize {image.n_batch}, currently {trans.n_batch}.")

    # Deal with the shape of input `image`
    n_batch = image.n_batch
    if n_batch == 1 and trans is not None and trans.n_batch is not None and trans.n_batch > 1: n_batch = trans.n_batch
    if n_batch == 1 and isinstance(target_space, bt.Tensor) and target_space.has_batch and target_space.n_batch > 1: n_batch = target_space.n_batch
    if image.n_batch == 1: image = image.repeated(n_batch, [])
    n_feature = image.n_channel
    size = bt.channel_tensor(image.space).int()
    if n_batch > 1 and not shape_out.has_batch: shape_out = bt.Size([n_batch]) + shape_out
    
    # Deal with input  `target_space`
    if target_space is None:
        scale, *pairs = trans.reshape
        if len(scale) == 1: scale *= n_dim
        target_space = [int(x * y) for x, y in zip(image.space, scale)]
        for p, q in pairs: target_space[p], target_space[q] = target_space[q], target_space[p]
        target_space = tuple(target_space)
        shape_out = shape_out.reset_space(target_space)
    if isinstance(target_space, tuple) and len(target_space) == n_dim: pass
    elif isinstance(target_space, bt.torch.Tensor): pass
    else: raise TypeError(f"Wrong target space for interpolation: {target_space}. ")
    if isinstance(target_space, tuple): 
        # Create a grid X with size ({n_dim}, size_1, size_2, ..., size_r) [r=n_dim].
        X = bt.image_grid(target_space).float() # + bt.channel_tensor([float(a-b)/2 for a, b in zip(image.space, target_space)])
        # Compute the transformed coordinates. Y: ([n_batch], {n_dim}, size_1, size_2, ..., size_r) [r=n_dim].
        if trans is None: trans = Identity()
        Y = trans(X)
        if not Y.has_batch: Y = Y.multiply(n_batch, [])
        if Y.n_batch == 1: Y = Y.repeated(n_batch, [])
        Y = Y.amplify(n_feature, [])
        shape_out = shape_out.reset_space(target_space)
    else:
        target_space = batorch_tensor(target_space)
        if not target_space.has_batch:
            if target_space.size(0) == n_batch and n_batch != n_dim or len([x for x in target_space.shape if x == n_dim]) >= 2:
                target_space.with_batch_dim(0)
            else: target_space = target_space.unsqueeze([])
        if not target_space.has_channel:
            if target_space.batch_dimension != 0 and target_space.size(0) == n_dim: target_space.with_channeldim(0)
            elif target_space.batch_dimension != 1 and target_space.size(1) == n_dim: target_space.with_channeldim(1)
            elif target_space.batch_dimension != target_space.n_dim - 1 and target_space.size(-1) == n_dim: target_space.with_channeldim(-1)
        avouch(target_space.has_channel and target_space.n_channel == n_dim, "'target_space' for interpolation should have a channel dimension for coordinates. ")
        Y = target_space.repeated(n_batch // target_space.n_batch, []).amplify(n_feature, [])
        shape_out = shape_out.reset_space(target_space.space)
        
    image = image.mergedims({}, [])
    n_batch = image.n_batch

    if method.lower() == 'bspline':
        if derivative: raise TypeError("No derivatives for bspline interpolations are available so far. Please write it by yourself. ")
        # TODO: FFD
        raise TypeError("Bspline interpolation is not available so far. Please write it by yourself. ")

    iY = bt.floor(Y).long() # Generate the integer part of Y
    # jY = iY + 1 # and the integer part plus one.
    if method.lower() == 'linear': fY = Y - iY.float() # The decimal part of Y.
    elif method.lower() == 'nearest': fY = bt.floor(Y - iY.float() + 0.5).long() # The decimal part of Y.
    else: raise TypeError("Unrecognized argument 'method'. ")
    # bY = bt.stack((iY, jY), 1).view([n_batch], 2, {n_dim}, -1) # ([n_batch], 2, {n_dim}, n_data).
    W = bt.stack((1 - fY, fY), 1).view([n_batch], 2, {n_dim}, -1) # ([n_batch], 2, {n_dim}, n_data).
    n_data = Y.flatten().size(-1)

    # Prepare for the output space: n_batch, m_1, ..., m_s
    if derivative: output = bt.zeros([n_batch], {n_dim}, *shape_out.space)
    else: output = bt.zeros(shape_out)
    
    for G in bt.image_grid([2]*n_dim).flatten().transpose(0, 1):
        ## Old version of interpolation with function 'gather'.
        # # Get the indices for the term: bY[:, G[D], D, :], size=([n_batch], {n_dim}, n_data)
        # Ind = bY.gather(1, G.expand_to([n_batch], 1, {n_dim}, n_data)).squeeze(1)
        Ind = (iY + G).flatten() # ([n_batch], {n_dim}, n_data).
        # Clamp the indices in the correct range & Compute the border condition
        condition = bt.sum((Ind < 0) + (Ind > size - 1), 1)
        Ind = bt.min(bt.clamp(Ind, min=0), (size - 1).expand_to(Ind))
        # Convert the indices to 1 dimensional. Dot: ([n_batch], n_data)
        Dot = Ind[:, 0]
        for r in range(1, n_dim): Dot *= size[r]; Dot += Ind[:, r]
        # Get the image values IV: ([n_batch], n_data)
        IV = None
        if isinstance(fill, str):
            if fill.lower() == 'nearest':
                IV = image.flatten().gather(1, Dot)
            elif fill.lower() == 'background':
                bk_value = bt.stack([image[(slice(None),) + tuple(g)] for g in (bt.image_grid([2]*n_dim) * bt.channel_tensor(size-1)).flatten().transpose(0,1)], 1).median(1).values
                background = bk_value * bt.ones_like(Dot)
            elif fill.lower() == 'zero':
                background = bt.zeros_like(Dot)
        else:
            background = fill * bt.ones_like(Dot)
        if IV is None: IV = bt.where(condition >= 1, background.float(), image.flatten().gather(1, Dot).float())
        # Weights for each point: [product of W[:, G[D], D, x] for D in range(n_dim)] for point x.
        # Wg: ([n_batch], {n_dim}, n_data)
        Wg = W.gather(1, G.expand_to([n_batch], 1, {n_dim}, n_data)).squeeze(1)
        if not derivative:
            output += (Wg.prod(1) * IV).view_as(output)
        else:
            tempWgMat = Wg.multiply(n_dim, 1) # ([n_batch], n_dim, {n_dim}, n_data)
            tempWgMat[:, bt.arange(n_dim), bt.arange(n_dim)] = 1
            dWg = tempWgMat.prod(1) * (G * 2 - 1).float()
            output += (dWg * IV.unsqueeze(1)).view_as(output)
    bt.torch.cuda.empty_cache()
    m = 0 if image.min() > -eps else image.min().item()
    M = 1 if image.max() < 1 + eps else image.max().item()
    return output.clamp(m, M)
