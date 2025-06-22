
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>.unittest",
    author = "Yuncheng Zhou", 
    create = "2023-07-03",
    fileinfo = "Unit test for static c++ version of dilation",
    requires = ["batorch", "pycamia", "micomputing", "unittest"]
)

import unittest
import os, sys, re
with __info__:
    import batorch as bt
    from pycamia import tokenize, scope, Workflow, execblock
    
workflow = Workflow()

class InheritTests(unittest.TestCase):
    
    def test_multivar_list(self):
        
        class MultiVariate(list):
            def __init__(self, *args, main_index=0, main_grad_index=1):
                if len(args) > 1: args = (list(args),)
                super().__init__(*args)
                self.main_index = main_index
                self.main_grad_index = main_grad_index
                
            for op in """
                __add__ __iadd__ __radd__
                __sub__ __isub__ __rsub__
                __mul__ __imul__ __rmul__
                __div__ __idiv__ __rdiv__
                __truediv__ __itruediv__ __rtruediv__
                __floordiv__ __ifloordiv__ __rfloordiv__""".split():
                execblock(f"""
                def {op}(self, other):
                    if isinstance(other, MultiVariate):
                        return MultiVariate((getattr(a, '{op}')(b) for a, b in zip(self, other)), main_index=self.main_index, main_grad_index=self.main_grad_index)
                    return MultiVariate((getattr(a, '{op}')(other) for a in self), main_index=self.main_index, main_grad_index=self.main_grad_index)
                """)
                    
            for op in """
                abs norm max min""".split():
                execblock(f"""
                def {op}(self): return MultiVariate((a.{op}() for a in self), main_index=self.main_index, main_grad_index=self.main_grad_index)
                """)
                    
            for prop in """
                shape""".split():
                execblock(f"""
                @property
                def {prop}(self): return [a.{prop} for a in self]
                """)
                    
            def cat(self): return bt.cat([a.flatten() for a in self])
                
            @property
            def main(self): return self[self.main_index]
                
            @main.setter
            def main(self, value): self[self.main_index] = value
                
            @property
            def main_grad(self): return self[self.main_grad_index]
                
            @main_grad.setter
            def main_grad(self, value): self[self.main_grad_index] = value
                
        print(0.1 * MultiVariate([bt.rand(4), bt.rand(5)]))
        
    def test_flattened_multivar(self):
        class MultiVariate(bt.Tensor):
	
            def __new__(cls, *args, main_index=0, main_grad_index=1):
                if len(args) > 1: args = list(args)
                else: args = list(args[0])
                
                assert len(args) > 0, "MultiVariate should at least be initialized by one variable. "
                
                match_props = dict(requires_grad=False, device=bt.device('cpu'), n_batch=None)
                match_casts = dict(requires_grad=lambda x, to: x.requires_grad_(to), device=lambda x, to: x.to(to), n_batch=lambda x, to: x if to is None else (x.repeated(to, {}) if x.has_batch else x.duplicate(to, {})))
                match_pivots = {}
                for arg in args:
                    for prop in match_props:
                        if hasattr(arg, prop):
                            value = getattr(arg, prop)
                            match_pivots.setdefault(prop, value)
                for prop, default in match_props.items():
                    match_pivots.setdefault(prop, default)
                    
                shapes = []
                tensor_starts = [0]
                for i, arg in enumerate(args):
                    for prop, default in match_props.items():
                        value = getattr(arg, prop, default)
                        casted = False
                        if not isinstance(arg, bt.Tensor):
                            arg = bt.tensor(arg); casted = True
                            if match_pivots[prop] is not None and value != match_pivots[prop]:
                                print(f"Warning: Variates of different '{prop}' detected in constructing MultiVariate, casting them to the first. ")
                                arg = match_casts.get(prop, lambda x, to: x)(arg, match_pivots[prop])
                                casted = True
                    if match_pivots['n_batch'] is None:
                        shape = arg.shape
                        args[i] = arg.flatten().init_special()
                    elif arg.has_batch:
                        arg = arg.movedim({}, 0)
                        shape = arg.shape
                        args[i] = arg.flatten(1).special_from(bt.Size({1}, 1))
                    else:
                        arg = arg.unsqueeze({})
                        shape = arg.shape.with_batch(match_pivots['n_batch'])
                        args[i] = arg.flatten(1).special_from(bt.Size({1}, 1))
                    tensor_starts.append(tensor_starts[-1] + args[i].size(-1))
                    shapes.append(shape)
                                
                self = super()._make_subclass(cls, bt.cat(args, -1)).special_from(args[0])
                self.inherited = dict(main_index=main_index, main_grad_index=main_grad_index, shapes=shapes, tensor_starts=tensor_starts[:-1])
                return self
                    
            def __str__(self):
                return super().__str__().rstrip(')').replace("Tensor", "MultiVariate") + f", tensor_shapes={self.inherited['shapes']})".replace("batorch.Size", "")
            
            def __repr__(self):
                return super().__repr__().replace("Tensor", "MultiVariate").replace(", requires_grad", f", tensor_shapes={self.inherited['shapes']}, requires_grad").replace("batorch.Size", "")
            
            def tuple(self):
                tensors = []
                for start, shape in zip(self.inherited['tensor_starts'], self.inherited['shapes']):
                    tensors.append(self[..., start:start+bt.tensor(shape[1:]).prod()])
                return tensors
            
            def get_range(self, index_name):
                index = self.inherited[index_name]
                start = self.inherited['tensor_starts'][index]
                shape = self.inherited['shapes'][index]
                return start, shape
            
            @property
            def main(self):
                start, shape = self.get_range('main_index')
                return self[..., start:start+bt.tensor(shape[1:]).prod()].view(shape)
                
            @main.setter
            def main(self, value):
                start, shape = self.get_range('main_index')
                if not shape.has_batch:
                    value = value.flatten().init_special()
                elif value.has_batch:
                    value = value.movedim({}, 0).flatten(1).special_from(bt.Size({1}, 1))
                else:
                    value = value.duplicate(shape.n_batch, {}).flatten(1).special_from(bt.Size({1}, 1))
                self[..., start:start+bt.tensor(shape[1:]).prod()] = value
                
            @property
            def main_grad(self):
                start, shape = self.get_range('main_grad_index')
                return self[..., start:start+bt.tensor(shape[1:]).prod()].view(shape)
                
            @main_grad.setter
            def main_grad(self, value):
                start, shape = self.get_range('main_grad_index')
                if not shape.has_batch:
                    value = value.flatten().init_special()
                elif value.has_batch:
                    value = value.movedim({}, 0).flatten(1).special_from(bt.Size({1}, 1))
                else:
                    value = value.duplicate(shape.n_batch, {}).flatten(1).special_from(bt.Size({1}, 1))
                self[..., start:start+bt.tensor(shape[1:]).prod()] = value
    
        MV = MultiVariate(bt.zeros({3}, 4).requires_grad_(True), bt.rand(2, 3))+0
        print(MV)
        print(MV + MV)
        print(MV.main)
        print(MV.tuple())
        MV.main_grad = bt.zeros(6)
        print(MV.tuple())
    
if __name__ == "__main__":
    unittest.main()