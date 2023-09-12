#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Author: Yuncheng Zhou
##############################

if __name__ == "__main__":
    import sys

    sys.path.append("../..")

    import torch
    import batorch as bt
    print(bt.__file__)
    from pycamia import scope
    import copy
    from batorch.device import AutoDevice

    # print(bt.stack(bt.zeros(3, 4), bt.ones(3, 4), dim={1}))

    # bt.set_autodevice(False)
    bt.set_device(bt.CPU)
    bt.manual_seed(0)
    with scope("test bt, cpu"):
    t = bt.randn([3000, 400], requires_grad=True)
    a = t
    LP = bt.nn.Linear(400, 400)
    for _ in range(10): a = LP(a)
    a.sum().backward()

    torch.manual_seed(0)
    with scope("test torch, cpu"):
    t_ = torch.randn([3000, 400], requires_grad=True)
    a_ = t_
    LP_ = torch.nn.Linear(400, 400)
    for _ in range(10): a_ = LP(a_)
    a_.sum().backward()

    assert t.allclose(t_)
    assert t._grad.allclose(t_._grad)


    with scope("test bt, gpu"):
        a = bt.Tensor(3000, 400, requires_grad=True)
        LP = bt.Tensor(400, 400, requires_grad=True)
        for t in range(10): a = a @ LP
        a.sum().backward()

    with scope("test torch, gpu"):
        a = torch.Tensor(3000, 400).requires_grad_().cuda()
        LP = torch.Tensor(400, 400).requires_grad_().cuda()
        for t in range(10): a = a @ LP
        a.sum().backward()

    with scope("test bt, gpu"):
        a = bt.Tensor(4000, 300, requires_grad=True)
        LP = bt.Tensor(300, 300, requires_grad=True)
        for t in range(10): a = a @ LP
        a.sum().backward()

    with scope("test torch, gpu"):
        a = torch.Tensor(4000, 300).requires_grad_().cuda()
        LP = torch.Tensor(300, 300).requires_grad_().cuda()
        for t in range(10): a = a @ LP
        a.sum().backward()
