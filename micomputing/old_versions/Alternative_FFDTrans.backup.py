

class FreeFormDeformation(Transformation):

    def __init__(self, offsets, spacing=1, origin=0):
        '''
        Free Form Deformation (FFD) transformation.
        
        Params:
            offsets [bt.Tensor]: the FFD offsets. 
                size: ([n_batch], {n_dim}, m_1, m_2, ..., m_r) 
                for m_1 x m_2 x ... x m_r [r=n_dim] grid of Δcontrol points
            spacing [int or tuple]: FFD spacing; spacing between FFD control points. 
            origin [int or tuple]: FFD origin; coordinate for the (0, 0, 0) control point. 
            
        When it is called:
            X [bt.Tensor]: Coordinates to be transformed.
                size: ([n_batch: optional], {n_dim}, n_1, n_2, ..., n_r)
            output [bt.Tensor]: The transformed coordinates.
                size: ([n_batch], {n_dim}, n_1, n_2, ..., n_r)
        '''
        if not isinstance(offsets, bt.Tensor): offsets = bt.tensor(offsets)
        if not offsets.has_channel:
            if offsets.size(0) == offsets.n_dim - 1:
                n_dim = offsets.size(0)
                offsets.channel_dimension = 0
                offsets = offsets.unsqueeze([])
            elif offsets.size(1) == offsets.n_dim - 2:
                n_dim = offsets.size(1)
                offsets.channel_dimension = 1
            else: raise TypeError(f"FFD parameters with size {offsets.shape} donot match ([n_batch], {{n_dim}}, m_1, m_2, ..., m_r) [r=n_dim]. ")
        if not offsets.has_batch:
            n_dim = offsets.n_channel
            if offsets.n_dim <= n_dim + 1: offsets = offsets.unsqueeze([])
            else: offsets.batch_dimension = 0
        avouch(offsets.has_batch and offsets.has_channel, f"Please use batorch tensor of size \
            ([n_batch], {{n_dim}}, m_1, m_2, ..., m_r) [r=n_dim] for FFD parameters, instead of {offsets.shape}. ")
        super().__init__(offsets, spacing, origin)
        n_dim = offsets.n_channel
        spacing = to_tuple(spacing)
        origin = to_tuple(origin)
        if len(spacing) == 1: spacing *= n_dim
        if len(origin) == 1: origin *= n_dim
        self.n_dim = n_dim
        self.offsets = offsets
        self.spacing = spacing
        self.origin = origin
    
    def __call__(self, X):
        X = super().__call__(X)
        n_dim = self.n_dim
        offsets = self.offsets.float()
        spacing = self.spacing
        n_batch = offsets.n_batch
        # Coord: size = ([n_batch], {n_dim}, n_1 * ... * n_r)
        Coord = X.flatten()
        Coord -= bt.channel_tensor(self.origin)
        tool = None
        if n_dim == 3: tool = FreeFormDeformation.FFD3D
        elif n_dim == 2: tool = FreeFormDeformation.FFD2D
        if tool:
            result = tool(Coord, offsets, spacing)
            result = result.view_as(X)
            result += bt.channel_tensor(self.origin)
            return result

    @staticmethod
    def term3D(a, b, c, ux, uy, uz, ix, iy, iz, offsets):
        # a/b/c: int; (u/i)(x/y/z): ([n_batch], {1}, n_data); offsets: ([n_batch], {n_dim=3}, mx, my, mz)
        mx, my, mz = offsets.space
        x, y, z = ix + a, iy + b, iz + c
        # deprecated code: 
        # offsets[b * Fones_like(x[b]), :, x[b], y[b], z[b]]
        # phi[b, d, k] = offsets[b, d, (x[b][k] * my + y[b][k]) * mz + z[b][k]]
        ind = ((x * my + y) * mz + z).repeated(3, {})
        ind = ind.clamp(0, mx * my * mz - 1).int()
        phi = offsets.flatten().gather(2, ind) # size: ([n_batch], {n_dim=3}, n_data)
        return (
            (0 <= x).float() * (x < mx).float() * Bspline(a * bt.ones_like(ux), ux) * 
            (0 <= y).float() * (y < my).float() * Bspline(b * bt.ones_like(uy), uy) * 
            (0 <= z).float() * (z < mz).float() * Bspline(c * bt.ones_like(uz), uz) * phi
        )

    @staticmethod
    def FFD3D(X, offsets, spacing):
        n_batch = offsets.n_batch # offsets: ([n_batch], {n_dim=3}, mx, my, mz)
        n_data = X.size(-1) # X: ([n_batch], {n_dim=3}, n_data)
        FFDX = X / bt.channel_tensor(spacing).float()
        iX = bt.floor(FFDX).float(); uX = FFDX - iX

        FFD = FreeFormDeformation
        args = uX.split() + iX.split() + (offsets,)
        delta = sum([
            FFD.term3D(-1, -1, -1, *args), FFD.term3D(-1, -1, 0, *args),
            FFD.term3D(-1, -1, 1, *args),  FFD.term3D(-1, -1, 2, *args),
            FFD.term3D(-1, 0, -1, *args),  FFD.term3D(-1, 0, 0, *args),
            FFD.term3D(-1, 0, 1, *args),   FFD.term3D(-1, 0, 2, *args),
            FFD.term3D(-1, 1, -1, *args),  FFD.term3D(-1, 1, 0, *args),
            FFD.term3D(-1, 1, 1, *args),   FFD.term3D(-1, 1, 2, *args),
            FFD.term3D(-1, 2, -1, *args),  FFD.term3D(-1, 2, 0, *args),
            FFD.term3D(-1, 2, 1, *args),   FFD.term3D(-1, 2, 2, *args),
            FFD.term3D(0, -1, -1, *args),  FFD.term3D(0, -1, 0, *args),
            FFD.term3D(0, -1, 1, *args),   FFD.term3D(0, -1, 2, *args),
            FFD.term3D(0, 0, -1, *args),   FFD.term3D(0, 0, 0, *args),
            FFD.term3D(0, 0, 1, *args),    FFD.term3D(0, 0, 2, *args),
            FFD.term3D(0, 1, -1, *args),   FFD.term3D(0, 1, 0, *args),
            FFD.term3D(0, 1, 1, *args),    FFD.term3D(0, 1, 2, *args),
            FFD.term3D(0, 2, -1, *args),   FFD.term3D(0, 2, 0, *args),
            FFD.term3D(0, 2, 1, *args),    FFD.term3D(0, 2, 2, *args),
            FFD.term3D(1, -1, -1, *args),  FFD.term3D(1, -1, 0, *args),
            FFD.term3D(1, -1, 1, *args),   FFD.term3D(1, -1, 2, *args),
            FFD.term3D(1, 0, -1, *args),   FFD.term3D(1, 0, 0, *args),
            FFD.term3D(1, 0, 1, *args),    FFD.term3D(1, 0, 2, *args),
            FFD.term3D(1, 1, -1, *args),   FFD.term3D(1, 1, 0, *args),
            FFD.term3D(1, 1, 1, *args),    FFD.term3D(1, 1, 2, *args),
            FFD.term3D(1, 2, -1, *args),   FFD.term3D(1, 2, 0, *args),
            FFD.term3D(1, 2, 1, *args),    FFD.term3D(1, 2, 2, *args),
            FFD.term3D(2, -1, -1, *args),  FFD.term3D(2, -1, 0, *args),
            FFD.term3D(2, -1, 1, *args),   FFD.term3D(2, -1, 2, *args),
            FFD.term3D(2, 0, -1, *args),   FFD.term3D(2, 0, 0, *args),
            FFD.term3D(2, 0, 1, *args),    FFD.term3D(2, 0, 2, *args),
            FFD.term3D(2, 1, -1, *args),   FFD.term3D(2, 1, 0, *args),
            FFD.term3D(2, 1, 1, *args),    FFD.term3D(2, 1, 2, *args),
            FFD.term3D(2, 2, -1, *args),   FFD.term3D(2, 2, 0, *args),
            FFD.term3D(2, 2, 1, *args),    FFD.term3D(2, 2, 2, *args),
        ])
        return delta + X

    @staticmethod
    
    def term2D(a, b, ux, uy, ix, iy, offsets):
        # a/b: int; (u/i)(x/y): ([n_batch], {1}, n_data); offsets: ([n_batch], {n_dim=2}, mx, my)
        mx, my = offsets.size()[2:]
        x, y = ix + a, iy + b
        ind = (x * my + y).repeated(2, {})
        ind = ind.clamp(0, mx * my - 1).int()
        phi = offsets.flatten().gather(2, ind) # size: ([n_batch], {n_dim=2}, n_data)
        return (
            (0 <= x).float() * (x < mx).float() * Bspline(a * bt.ones_like(ux), ux) * 
            (0 <= y).float() * (y < my).float() * Bspline(b * bt.ones_like(uy), uy) * phi
        )

    @staticmethod
    
    def FFD2D(X, offsets, spacing):
        n_batch = offsets.n_batch # offsets: ([n_batch], {n_dim=2}, mx, my)
        n_data = X.size(-1) # X: ([n_batch], {n_dim=2}, n_data)
        FFDX = X / bt.channel_tensor(spacing)
        iX = bt.floor(FFDX).float(); uX = FFDX - iX

        FFD = FreeFormDeformation
        args = uX.split() + iX.split() + (offsets,)
        delta = sum([
            FFD.term2D(-1, -1, *args), FFD.term2D(-1, 0, *args),
            FFD.term2D(-1, 1, *args),  FFD.term2D(-1, 2, *args),
            FFD.term2D(0, -1, *args),  FFD.term2D(0, 0, *args),
            FFD.term2D(0, 1, *args),   FFD.term2D(0, 2, *args),
            FFD.term2D(1, -1, *args),  FFD.term2D(1, 0, *args),
            FFD.term2D(1, 1, *args),   FFD.term2D(1, 2, *args),
            FFD.term2D(2, -1, *args),  FFD.term2D(2, 0, *args),
            FFD.term2D(2, 1, *args),   FFD.term2D(2, 2, *args),
        ])
        return delta + X
