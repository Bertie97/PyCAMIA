
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    import copy
    import torch
    import numpy as np
    sys.path = ["../.."] + sys.path


    import batorch as bt
    from pycamia import scope

    t = bt.randn(3000, 400, requires_grad=True)
    print(bt.nn.functional.normalize(t, p=2, dim=1))

    bt.turn_on_autodevice()
    bt.manual_seed(0)
    with scope("test tp, gpu"):
        t = bt.randn(3000, 400, requires_grad=True)
        a = t
        LP = bt.nn.Linear(400, 400)
        for _ in range(10): a = LP(a).relu()
        a.sum().backward()

    torch.manual_seed(0)
    with scope("test torch, gpu"):
        t_ = bt.get_device()(torch.randn(3000, 400)).requires_grad_(True)
        a_ = t_
        LP_ = bt.get_device()(torch.nn.Linear(400, 400))
        for _ in range(10): a_ = LP_(a_).relu()
        a_.sum().backward()

    assert a.is_cuda is True
    assert t.allclose(t_)
    assert isinstance(t, bt.Tensor)
    assert isinstance(a, bt.Tensor)
    assert isinstance(LP.weight, bt.nn.Parameter)
    assert isinstance(LP.bias, bt.nn.Parameter)
    assert isinstance(bt.tensor(np.array([1., 2.])), bt.Tensor)
    if torch.cuda.is_available():
        assert a.is_cuda
        assert t.is_cuda
        assert bt.tensor(np.array([1., 2.])).is_cuda

    bt.set_autodevice(False)
    bt.manual_seed(0)
    with scope("test tp, cpu"):
        t = bt.randn(3000, 400, requires_grad=True)
        a = t
        LP = bt.nn.Linear(400, 400)
        for _ in range(10): a = LP(a).relu()
        a.sum().backward()

    torch.manual_seed(0)
    with scope("test torch, cpu"):
        t_ = torch.randn(3000, 400).requires_grad_()
        a_ = t_
        LP_ = torch.nn.Linear(400, 400)
        for _ in range(10): a_ = LP_(a_).relu()
        a_.sum().backward()

    assert a.is_cuda is False
    assert t.allclose(t_)
    assert isinstance(t, bt.Tensor)
    assert isinstance(a, bt.Tensor)
    assert isinstance(LP.weight, bt.nn.Parameter)
    assert isinstance(LP.bias, bt.nn.Parameter)
    assert isinstance(bt.tensor(np.array([1., 2.])), bt.Tensor)

    bt.nn.ParameterList([bt.nn.Parameter(bt.zeros(30)), bt.nn.Parameter(bt.zeros(30))])
    bt.nn.ParameterList([LP.weight, LP.bias])
