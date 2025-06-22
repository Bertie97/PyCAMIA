
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>.unittest",
    author = "Yuncheng Zhou", 
    create = "2023-08",
    fileinfo = "Unit test for file copying.",
    requires = ["pycamia", "unittest"]
)

import unittest
import os, sys, re, math
with __info__:
    from pyoverload import int
    from pycamia import *
workflow = Workflow("pycamia copy folder")

class ArgumentTests(unittest.TestCase):
    
    def test_arguments(self):
        with Args() as args:
    
            # data options
            args.dataset        ["The dataset used for the experiment. "]                        = "ADNI"
            args.reorient       [bool: "Whether the data are reoriented, should be set "
                + "(to False) for skewed data such as cardiac data. "]      = True
            args.train_prop     ["The proportion of the traing dataset. "]                       = 0.75
            args.val_prop       ["The proportion of the validation set. "]                       = 0.1
            with args('-c', '--crop'):
                args.crop       ["The function that computes the crop size (in Python). "]       = 'lambda x: x'
            # basic options
            args.manual_seed    ["The manual seed for the randomness. "]                         = 1234
            args.n_dim          [int[3]:"The number of data dimension. "]                               = 3
            args.n_batch        ["The batch size. "]                                             = 0 # Data-specific
            args.n_epoch        ["The number of epochs for training. "]                          = 0 # Data-specific
            
            # evaluation options
            args.phase          ["The running phase. Options: train, eval, continue"]            = 'train'
            args.timestamp      ["The timestamp of the evaluated checkpoint. "]                  = '202409041729'
            args.i_epoch        ["The index of the epoch used for model evaluation."]            = 'best'
            args.mesh_gap       ["The gap between lines of the mesh. "]                          = 6
            args.labels         ["The label for evaluation, in the format: name[label value]. "
                +"Multiple outputs are seperated by '/'. Use union, all, each "
                +"for the union of labels, the average of all labels and "
                +"all labels named after the label values. "
                +"e.g. 'union/all/LV[500]/RV[600]/Myo[200]'. "]                  = '' # Data-specific
            args.eval           ["'--eval xxx' is the alias of '--phase eval --timestamp xxx'"]  = ''
            if args.eval: args.phase = 'eval'; args.timestamp = args.eval

            # network structure
            args.method         ["The ODE method chosen. Options: UNODE, SNODE, CNODE, RCNODE"]  = 'UNODE'
            args.n_layer        ["The number of layers (-1 for auto). "]                         = 20
            args.tolerance      ["The tolerance value. "]                                        = 0
            args.block_channels ["The number of feature channels. "]                             = 16
            args.mid_channels   ["The number of middel channels. "]                              = 16
            args.depth          ["The depth of encoder Unet. "]                                  = 4
            args.down_scale     ["The down scaling factor"]                                      = 2

            # training parameters
            args.learning_rate  ["Learning rate (or i_iter->lr func). "]                         = 0. # Data-specific
            args.loss           ["The loss used to train the network. Options: NMI, NVI, MSE"]   = 'NMI'
            args.bend_coeff     ["The coefficient of bending energy. "]                          = 0. # Data-specific
            args.auto_grad      ["Whether to use auto grad for ODE. "]                           = False
            args.epoch_save     ["The number of epochs before recursively saving the model."]    = 10

            # transformation settings
            args.affine_scale   ["The scaling for the non-identical values in affine matrix."]   = 0.05
            args.velocity_scale ["The scaling for the velocity fields. "]                        = 4
            args.velocity_type  ["The backbone for velocity model. Options: DDF, FFD"]           = 'FFD'
            args.ffd_spacing    ["The spacing between adjacent FFD control points (in px). "]    = 10

            print(args)
        
        
if __name__ == "__main__":
    unittest.main()