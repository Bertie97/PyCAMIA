
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>.unittest",
    author = "Yuncheng Zhou", 
    create = "2023-08-20",
    fileinfo = "Unit test for medical image transformation",
    requires = ["pycamia", "micomputing", "unittest"]
)

import unittest
import os, sys, re, random
with __info__:
    import batorch as bt
    import micomputing as mc
    from micomputing.zxhtools.TRS import TRS
    from pycamia import scope, Path, Workflow
    
workflow = Workflow()
pwd = Path("UnitTests")/"Transformation_Resources"
# pwd = Path("Transformation_Resources")

class TransformationTests(unittest.TestCase):
    
    def __init__(self, *args):
        super().__init__(*args)
        self.source = mc.IMG(pwd/"input_data"/"16_fat.nii.gz")
        self.target = mc.IMG(pwd/"input_data"/"15_opp.nii.gz")
        self.transformation = TRS.load(pwd/"input_data"/"16fat_on_15opp.AFF").trans.to_image_space(self.source, self.target)
        self.golden_standard = mc.IMG(pwd/"input_data"/"16fat_on_15opp.nii.gz")

    def test_backward_transformation(self):
        with scope("backward transformation"):
            transformed = mc.interpolation(self.source.to_tensor(), self.transformation).int()
        self.target.save(transformed, pwd/"output_data"/"test_backward.nii.gz")
    
    def test_forward_transformation(self):
        with workflow("use FFD", "use DDF", "use Id"), workflow.any_tag:
            if "use FFD" in workflow.workflow:
                transformation = mc.FreeFormDeformation(bt.randn([1], {3}, *[x//2 for x in source.shape]), spacing=2).fake_inv()
            elif "use DDF" in workflow.workflow:
                transformation = mc.DenseDisplacementField(bt.randn([1], {3}, *source.shape)).fake_inv()
            elif "use Id" in workflow.workflow:
                transformation = mc.Identity()
        self.transformation.backward_()
        t = self.transformation.force_inv(*self.source.shape) @ self.transformation
        with workflow("show transformation grid"), workflow.use_tag:
            mc.plt.gridshow(t, on=self.source.to_tensor(), gap=5, as_orient = "PIL", to_orient='RS')
            mc.plt.show()
            return
        with scope("forward transformation"):
            transformed = mc.interpolation_forward(self.source.to_tensor(), self.transformation, target_space=self.target.to_tensor().space, sigma=0.33).int()
        self.target.save(transformed, pwd/"output_data"/"test_forward.nii.gz")
    
    def test_inverse_transformation(self):
        transformation = mc.DenseDisplacementField(self.transformation.inv().to_DDF(*self.source.shape)).force_inv(*self.target.shape).backward_()
        with scope("double inverse transformation"):
            transformed = mc.interpolation(self.source.to_tensor(), transformation).int()
        self.target.save(transformed, pwd/"output_data"/"test_inverse.nii.gz")
    
    def test_save_as_nii(self):
        self.transformation.save_as_nii(pwd/"output_data"/"16fat_on_15opp.aff.nii.gz", target=self.target)

class TransformationSubClassMethodTests(unittest.TestCase):

    def test_inv_method(self):
        n_dim = 3
        size = bt.randint[50, 100](n_dim).tolist()
        sp = tuple((1e-1 * bt.randn(n_dim) + 1).tolist())
        for trans_name in dir(mc):
            trans_type = getattr(mc, trans_name)
            if isinstance(trans_type, type) and issubclass(trans_type, mc.SpatialTransformation) and hasattr(trans_type, 'inv'):
                X = bt.image_grid(*size)
                if hasattr(trans_type, 'random_init_params'):
                    params = trans_type.random_init_params(*size)
                elif trans_name.startswith('Id'):
                    params = tuple()
                elif trans_name.startswith('Rotation') or trans_name.startswith('Reflect'):
                    params = (0, 1)
                elif 'Permut' in trans_name:
                    dims = list(range(len(size)))
                    random.shuffle(dims)
                    params = tuple(dims)
                else:
                    print(f"[ Warning ]: {trans_name} has 'inv' but no param initializer, thus was skipped. ")
                    continue
                try: trans = trans_type(*params, trans_stretch=2, spacing=sp)
                except TypeError:
                    try: trans = trans_type(*params, trans_stretch=2)
                    except TypeError:
                        try: trans = trans_type(*params, spacing=sp)
                        except TypeError: trans = trans_type(*params)
                # trans = trans_type(*params)
                Y = trans.inv()(trans(X))
                error = bt.norm(X - Y).mean().mean().item()
                if error < 1:
                    print(f"[Succeeded]: {trans_name} checked, finding an error of {error}.")
                else:
                    print(f"[ Failed  ]: {trans_name} checked, finding an error of {error}.")

    def test_save_load_composed(self):
        size = bt.randint[20, 40](3).tolist()
        X = bt.image_grid(*size)
        trans = mc.ComposedTransformation(
            mc.Affine(mc.Affine.random_init_params(*size)), 
            mc.DenseDisplacementField(mc.DenseDisplacementField.random_init_params(*size))
        )
        Y = trans(X)
        trans.save(pwd/"trans_files"/"composed.trs")
        read_trans = mc.Transformation.load(pwd/"trans_files"/"composed.trs")
        Z = read_trans(X)
        error = bt.norm(Y - Z).mean().mean().item()
        print(f"Error = {error}")

    def test_save_load_MLP(self):
        size = bt.randint[50, 100](3).tolist()
        X = bt.image_grid(*size)
        trans = mc.MultiLayerPerception(mc.MultiLayerPerception.random_init_params(len(size), [10, 10]), hidden_layers=[10, 10])
        Y = trans(X)
        trans.save(pwd/"trans_files"/"MLP.trs")
        read_trans = mc.Transformation.load(pwd/"trans_files"/"MLP.trs")
        Z = read_trans(X)
        error = bt.norm(Y - Z).mean().mean().item()
        print(f"Error = {error}")

if __name__ == "__main__":
    unittest.main()
