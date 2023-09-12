
if __name__ == "__main__":
    import sys
    sys.path.append("../..")
    import torch

    import os
    os.environ['CUDA_RUN_MEMORY'] = '2'
    import batorch as bt
    from pycamia import scope

    ##############################
    ## Test CPU
    ##############################

    with scope("torch, cpu, cat"):
        torch.cat([torch.zeros(300, 300), torch.zeros(300, 300)])

    with scope("bt, cpu, cat"):
        bt.cat([bt.zeros(300, 300), bt.zeros(300, 300)])
