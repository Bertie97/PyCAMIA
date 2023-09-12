
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "micomputing",
    author = "Yuncheng Zhou",
    create = "2022-02",
    fileinfo = "File to format transformation files from `zxhtools`.",
    requires = ["numpy", "batorch", "pycamia"]
).check()

__all__ = """
    AFF FFD DDF FLD TRS
    transform
    zxhtransform
""".split()

from ..trans import Affine, DenseDisplacementField, FreeFormDeformation
from ..trans import ComposedTransformation, Quaterns2Matrix, interpolation
from ..stdio import IMG

with __info__:
    import numpy as np
    import batorch as bt
    from pycamia import SPrint, alias, Path

class TRS:

    def __init__(self, **params):
        self.__dict__.update(params)
        self.path = None

    @classmethod
    def load(cls, p):
        p = Path(p)
        with p.open() as fp:
            self = cls.parse(fp.read())
            self.path = Path(p)
            return self

    @classmethod
    def parse(cls, string):
        blocks = string.strip('\n#').split('\n#')
        variables = [tuple(x.strip() for x in b.split('#')) for b in blocks]
        var_names = [x[0].replace(' ', '_').replace(',', '__c__') for x in variables]
        variables = dict(variables)
        params = {}
        for var in variables:
            value = variables[var]
            try:
                value = [[[[eval(d.strip()) for d in c.strip().split(' ')] for c in b.strip().split('\t')] for b in a.strip().split('\n')] for a in value.strip().split('\n\n')]
                value = np.array(value).squeeze()
            except (NameError, SyntaxError): pass
            if isinstance(value, str):
                value = value.replace('\\', '/')
            params[var.replace(' ', '_').replace(',', '__c__')] = value
        if cls == TRS:
            if any(['matri' in k for k in params.keys()]): cls = AFF
            elif any(['control_point' in k for k in params.keys()]): cls = FFD
            elif any(['field' in k for k in params.keys()]): cls = FLD
        return cls(**params, var_names=var_names)

    def __str__(self):
        output = SPrint()
        for key in self.var_names:
            name = key.replace('__c__', ',').replace('_', ' ')
            output("#{}#".format(name))
            value = getattr(self, key)
            if isinstance(value, str): output(value)
            if value.ndim == 0: output(value)
            if value.ndim == 1: output(*[self.pattern.get(name, {}).get('format', '%d')%v for v in value], sep=self.pattern.get(name, {}).get('sep', '\n'))
            if value.ndim == 2: output(*['\t'.join([self.pattern.get(name, {}).get('format', '%f')%v for v in c]) for c in value.tolist()], sep='\n')
            if value.ndim > 2: output(*['\n'.join(['\t'.join([' '.join([self.pattern.get(name, {}).get('format', '%f')%v for v in c]) for c in b]) for b in a]) for a in value.tolist()], sep='\n\n')
        return output.text

    def save(self, p):
        p = Path(p)
        with p.open('w') as fp:
            fp.write(str(self))

    def get(self, key):
        return getattr(self, key.replace(' ', '_').replace(',', '__c__'))
    
    def to_image_space(self, source, target):
        if isinstance(source, str): source = IMG(source).affine
        if isinstance(target, str): target = IMG(target).affine
        return self.trans.to('image', source_affine=source, target_affine=target)
    
class AFF(TRS):

    pattern = {
        'matrix': {'format': '%.6f'},
        'skew in x-y and y-x plane': {'format': '%.6f'},
        'skew in y-z and z-y plane': {'format': '%.6f'},
        'skew in z-x and x-z plane': {'format': '%.6f'},
        'rotation center': {'sep': '\t', 'format': '%.6f'},
        'rotation angle': {'sep': '\t', 'format': '%.6f'},
        'scaling': {'sep': '\t', 'format': '%.6f'},
        'translation': {'sep': '\t', 'format': '%.6f'}
    }
    
    @property
    def trans(self):
        if not hasattr(self, '_trans'):
            self._trans = Affine(bt.tensor(self.matrix).float())
        return self._trans

class FFD(TRS):

    pattern = {
        'control grid spacing': {'format': '%15.6f'},
        'offsets of the control points': {'format': '%.6f'},
        'quaternion qf,qb,qc,qd,qx,qy,qz': {'sep': '\t', 'format': '%.6f'}
    }
    
    @property
    def trans(self):
        if not hasattr(self, '_trans'):
            qf,qb,qc,qd,qx,qy,qz = self.get('quaternion qf,qb,qc,qd,qx,qy,qz')
            affmat = Quaterns2Matrix([qb,qc,qd,qx,qy,qz]).float()
            if qf == -1: affmat = affmat @ bt.diag([1., 1., -1., 1.])
            offsets = bt.tensor(self.offsets_of_the_control_points).float().T.channel_dimension_(0)
            # grid_spacing = (bt.inv(affmat)[0, :3, :3] @ bt.tensor(self.control_grid_spacing).float().unsqueeze(-1)).abs().squeeze().tolist()
            ffd = FreeFormDeformation(offsets, spacing = self.control_grid_spacing)
            self._trans = ComposedTransformation(Affine(affmat), ffd, Affine(bt.inv(affmat)), mode="spatial")
        return self._trans

@alias('DDF')
class FLD(TRS):

    pattern = {}
    
    @property
    def trans(self):
        if not hasattr(self, '_trans'):
            pathx = Path(self.get('transformation fields in dimension x'))
            pathy = Path(self.get('transformation fields in dimension y'))
            pathz = Path(self.get('transformation fields in dimension z'))
            if self.path is not None:
                pathx = self.path.parent/pathx.filename
                pathy = self.path.parent/pathy.filename
                pathz = self.path.parent/pathz.filename
            dx = IMG(pathx).to_tensor()
            dy = IMG(pathy).to_tensor()
            dz = IMG(pathz).to_tensor()
            disp = bt.stack(dx, dy, dz, {})
            self._trans = DenseDisplacementField(disp)
        return self._trans
    
    @property
    def space(self):
        if not hasattr(self, '_disp'):
            pathx = Path(self.get('transformation fields in dimension x'))
            pathy = Path(self.get('transformation fields in dimension y'))
            pathz = Path(self.get('transformation fields in dimension z'))
            if self.path is not None:
                pathx = self.path.parent/pathx.filename
                pathy = self.path.parent/pathy.filename
                pathz = self.path.parent/pathz.filename
            disp = bt.stack(IMG(pathx).to_tensor(), IMG(pathy).to_tensor(), IMG(pathz).to_tensor(), {})
            self._disp = disp
        return self._disp
    
def transform(target, source, output=None, *trans, **kwargs):
    if output is None: output = Path(source)^target.name.split('_')[-1]
    target_img = IMG(target)
    all_trans = []
    for t in trans:
        x = TRS.load(t)
        if isinstance(x, FLD):
            if 'SIC' in str(t) or 'SIC' in x.get('transformation fields in dimension x'):
                sp = x.space
                all_trans.append(DenseDisplacementField(sp - bt.image_grid(sp.space)).to_world_space(source, target))
                continue
        all_trans.append(x.trans)
    target_img.save(interpolation(IMG(source), ComposedTransformation(*all_trans).to_image_space(source, target), target_space=target_img.shape, **kwargs), output)
    
def zxhtransform(target, source, *trans):
    return transform(target, source, *trans[::-1])


if __name__ == '__main__':
    aff = AFF.load('image2.AFF')
    ffd = FFD.load('image2.FFD')
    print(aff, ffd)
