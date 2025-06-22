
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "Optimizers. ",
    requires = "torch"
)

__all__ = """
    CSGD
    CADAM
    Optimization
    train
    test
""".split()

import math
from functools import wraps
from typing import Generator

with __info__:
    import batorch as bt
    from pycamia import alias, avouch
    
class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

class CSGD(bt.optim.Optimizer):
    
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)
        self.DGDG = bt.tensor([])
        self.DFDG = bt.tensor([])

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            
    def minimize(self, loss, **kwargs):
        self.DGDG = bt.tensor([])
        self.DFDG = bt.tensor([])
        self.zero_grad()
        loss.backward(**self.kwargs)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = bt.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                param_state['loss_grad'] = bt.clone(d_p.data).detach()
                param_state['cons_grad'] = []
        return self
        
    def under_constraint(self, cons):
        self.zero_grad()
        cons.backward()
        innerprods = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if 'cons_grad' not in param_state: param_state['cons_grad'] = []
                param_state['cons_grad'].append(bt.clone(p.grad.data).detach())
                if innerprods is None: innerprods = bt.zeros(len(param_state['cons_grad']) + 1)
                innerprods[0] += (param_state['loss_grad'] * param_state['cons_grad'][-1]).sum().detach()
                innerprods[1:] += (bt.stack(param_state['cons_grad']) * bt.unsqueeze(param_state['cons_grad'][-1])).flatten(1).sum(1).detach()
        self.DFDG = bt.cat((self.DFDG, bt.unsqueeze(innerprods[0])))
        self.DGDG = bt.cat((bt.cat((self.DGDG, innerprods[1:-1].view(1, -1))), innerprods[1:].view(-1, 1)), 1)
        return self

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        if len(self.DFDG) > 0: lagrange = bt.inverse(self.DGDG) @ self.DFDG.view(-1, 1)
        else: lagrange = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                
                d_p = param_state['loss_grad']
                for i, l in enumerate(lagrange):
                    d_p.data.add_(param_state['cons_grad'][i], alpha=-l.item())

                p.data.add_(d_p, alpha=-group['lr'])

        return loss
        
class CADAM(bt.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        self.cons_values = []
        self.cons_strength = []
        self.DGDG = None
        self.DFDG = None
        self.stepped = False
        self.adjusted = False

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            
    def minimize(self, loss, **kwargs):
        self.kwargs = kwargs
        self.zero_grad()
        self.DGDG = None
        self.DFDG = None
        loss.backward(**self.kwargs)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = bt.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = bt.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = bt.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    bt.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    
                state['loss_grad'] = (bt.clone((exp_avg / denom).data).detach() / bias_correction1).special_from_(grad)
                state['cons_grad'] = []
        self.stepped = False
        self.adjusted = False
        self.cons_values = []
        self.cons_strength = []
        return self
        
    def under_constraints(self, cons, strength = 1, **kwargs):
        kwargs.pop('retain_graph', None)
        for i, c in enumerate(cons): self.under_constraint(c, strength=strength, retain_graph = i < len(cons) - 1, **kwargs)
        return self
        
    def under_constraint(self, cons, strength = 1, **kwargs):
        assert 0 <= strength <= 1
        self.zero_grad()
        self.cons_values.append(cons)
        avouch(cons.n_space_dim == 0)
        if cons.has_batch: n_batch = cons.n_batch
        else: n_batch = None
        cons.sum().backward(**kwargs)
        innerprods = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                param_state = self.state[p]
                if 'cons_grad' not in param_state: param_state['cons_grad'] = []
                param_state['cons_grad'].append(p.grad.detach())
                if innerprods is None:
                    if n_batch is None: innerprods = bt.zeros(len(param_state['cons_grad']) + 1)
                    else: innerprods = bt.zeros([n_batch], len(param_state['cons_grad']) + 1)
                innerprods[..., 0] += (param_state['loss_grad'] * param_state['cons_grad'][-1]).sum().detach()
                DG_x_DG = bt.stack(param_state['cons_grad']) * bt.unsqueeze(param_state['cons_grad'][-1])
                if DG_x_DG.has_batch:
                    if DG_x_DG.ndim > 2: DG_x_DG = DG_x_DG.flatten(2).sum(2)
                    DG_x_DG = DG_x_DG.transpose(0, 1)
                elif DG_x_DG.ndim > 1: DG_x_DG = DG_x_DG.flatten(1).sum(1)
                innerprods[..., 1:] += DG_x_DG.detach()
        if self.DFDG is None: self.DFDG = innerprods[..., :1]
        else: self.DFDG = bt.cat(self.DFDG, innerprods[..., :1], -1)
        if self.DGDG is None: self.DGDG = innerprods[..., 1:].unsqueeze(-1)
        else:
            row_added = bt.cat(self.DGDG, innerprods[..., 1:-1].unsqueeze(-2), -2)
            self.DGDG = bt.cat(row_added, innerprods[..., 1:].unsqueeze(-1), -1)
        self.cons_strength.append(strength)
        return self

    def step(self, closure=None):
        self.stepped = True
        loss = None
        if closure is not None:
            loss = closure()
            
        if self.DFDG is not None:
            self.DGDG += 1e-6 * (bt.det(self.DGDG) <= 1e-6) * bt.eye_as(self.DGDG)
            lagranges = (bt.inverse(self.DGDG) @ self.DFDG.unsqueeze(-1)).squeeze(-1).split(1, -1)
        else: lagranges = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                
                d_p = param_state['loss_grad']
                for i, l in enumerate(lagranges):
                    d_p.data.add_(l * param_state['cons_grad'][i], alpha = -self.cons_strength[i])

                p.data.add_(d_p, alpha=-group['lr'])

        return loss
        
    def adjust(self, *G, to_zero = False):
        if not self.stepped: raise TypeError("step not called")

        if len(G) > 0: eps = (bt.inverse(self.DGDG) @ (bt.stack(G, -1).unsqueeze(-1) - bt.stack(self.cons_values, -1).unsqueeze(-1) * int(not to_zero))).squeeze(-1)
        else: eps = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                param_state = self.state[p]
                for i, e in enumerate(eps.split(1, -1)):
                    p.data.add_(-e * self.cons_strength[i] * param_state['cons_grad'][i])

def strify(x):
    if isinstance(x, bt.torch.Tensor):
        if x.numel() == 1: return x.item()
        else: return x.tolist()
    return x

class Optimization:
    
    def __init__(self, optimizer, params_or_model, dataset = None, lr = 1e-3, **kwargs):
        if isinstance(params_or_model, bt.nn.Module):
            init_params = params_or_model.parameters()
            self.model = params_or_model
            self.dataset = dataset
        else:
            init_params = params_or_model
            self.model = None
            self.dataset = dataset
        self.hyper = dict(lr = lr if callable(lr) else lambda _: lr)
        self.hyper.update({k: v if callable(v) else lambda _: v for k, v in kwargs.items()})
        if isinstance(init_params, (Generator, tuple)): init_params = list(init_params)
        if not isinstance(init_params, list): init_params = [init_params]
        if not isinstance(init_params[0], dict) or 'params' not in init_params[0]:
            init_params = [dict(params = x, **{k: v(0) for k, v in self.hyper.items()}) for x in init_params]
        self._optimizer = optimizer(init_params)
        self._index = 1
        self.loss = []
        self.loss_record = {}

    def optimizer(self, i):
        # state_dict = self._optimizer.state_dict()
        # # for param_state_dict in state_dict['param_groups']:
        # #     for k in self.hyper:
        # #         param_state_dict[k] = self.hyper[k](i)
        # #         print(f"{k}: {self.hyper[k](i)}")
        # self._optimizer.load_state_dict(state_dict)
        for param_group in self._optimizer.param_groups:
            for k in self.hyper:
                param_group[k] = self.hyper[k](i)
        return self._optimizer
    
    def new_iter(self):
        optimizer = self.optimizer(self._index)
        self._index += 1
        return optimizer
            
    def minimize(self, loss, **kwargs):
        """Minimize the loss function in the optimizer.

        Args:
            loss (bt.Tensor or callable): The loss funtion, a batorch tensor with graph
                or a function taking arguments (model, batch_data).
            update_lr (bool): Whether update the learning rate. Defaults to True. 
            Note: batch_data is obtained by calling method 'training_batch(n_batch)' in optimizer.dataset.
        """
        if callable(loss): self.loss.append((1, loss)); return
        optimizer = self.optimizer(self._index)
        optimizer.zero_grad()
        loss.backward(**kwargs)
        optimizer.step()
        self._index += 1

    def maximize(self, loss, **kwargs):
        """Maximize the loss function in the optimizer.

        Args:
            loss (bt.Tensor or callable): The loss funtion, a batorch tensor with graph
                or a function taking arguments (model, batch_data).
            Note: batch_data is obtained by calling method 'training_batch(n_batch)' in optimizer.dataset.
        """
        if callable(loss): self.loss.append((-1, loss)); return
        optimizer = self.optimizer(self._index)
        optimizer.zero_grad()
        (-loss).backward(**kwargs)
        optimizer.step()
        self._index += 1

    @alias("optimize")
    def train(self, max_epoch=10, n_epoch_valid=1, n_batch=4):
        """
        Optimize the function [train the model] by this optimizer. 
        Warning: Requires Dataset element in package `micomputing`. 

        Args:
            max_epoch (int): The number of times we scan the dataset, i.e. the number of epochs. Defaults to `10`.
            n_epoch_valid (int): Validate every n_epoch_valid epochs. Defaults to `1`.
            n_batch (int): Batch size for training and validation. Defaults to `4`.
        """
        avouch(self.dataset is not None and self.model is not None, 
               "Model and dataset needed for training an optimization problem. ")
        epoch_strlen = len(str(max_epoch))
        for i_epoch in range(1, max_epoch + 1):
            epoch = f'%0{epoch_strlen}d'%i_epoch
            i = 1
            while True:
                self.model.train()
                batch = self.dataset.training_batch(n_batch)
                if batch is None: break
                loss = self.compute_loss(self.model, batch)
                self.minimize(loss)
                print(f"[TRAIN] epoch = {epoch}, iteration = {'%04d'%i}, loss = {strify(loss)}")
                i += 1
            if i_epoch % n_epoch_valid == 0:
                self.model.eval()
                batch = self.dataset.validation_batch(n_batch, restart = True)
                if batch is None: break
                loss = self.compute_loss(self.model, batch)
                print(f"[VALID] epoch = {epoch}, loss = {strify(loss)}")

    def compute_loss(self, model, batch):
        total_loss = bt.tensor(0)
        for d, l in self.loss:
            l_value = l(model, batch)
            self.loss_record[l.__name__] = l_value
            total_loss += d * l_value
        return total_loss

    def get_loss(self, name):
        return self.loss_record.get(name, None)

    def __getitem__(self, i): return self.optimizer(i)

def train(dataset, max_epoch=10, n_epoch_valid=1, n_batch=4, iter_in_epoch=None, valid_info=None):
    """
    Train by dataset. 
    Warning: Requires `Dataset` in package `micomputing`. 
    
    Note: Please use a function for loss computation (see example). It takes index and data batch. 
        index is a tuple of 3, including epoch index, iteration in the current epoch and the total iterations. 
        The latter two are 0s for validation.

    Args:
        dataset (Dataset): The dataset structure.
        max_epoch (int): The number of times we scan the dataset, i.e. the number of epochs. Defaults to `10`.
        n_epoch_valid (int): Validate every n_epoch_valid epochs. Defaults to `1`.
        n_batch (int): Batch size for training and validation. Defaults to `4`.
        valid_info (None or list): The manually specified batch to validate.
    
    Examples::
        >>> @train(dataset)
        ... def loss(index, batch_data):
        ...     stage, epoch, iteration, total_iter = index
        ...     # stage: 'training' / 'validation'; iteration: i_iter in current epoch
        ...     model1.train()
        ...     loss1 = loss_func(batch_data)
        ...     opt1.minimize(loss1)
        ...     if total_iter % 100 == 0:
        ...         loss2 = loss_func(batch_data)
        ...         opt2.maximize(loss2)
        ... return {'loss1': loss1, 'loss2': loss2}
    """
    def decorator(func):
        k = 1
        epoch_strlen = len(str(max_epoch))
        for i_epoch in range(1, max_epoch + 1):
            epoch = f'%0{epoch_strlen}d'%i_epoch
            i = 1
            while True:
                batch = dataset.training_batch(n_batch)
                if batch is None: break
                if iter_in_epoch is not None and i > iter_in_epoch: break
                results = func(('training', i_epoch, i, k), batch)
                if not isinstance(results, dict): results = {'loss': results}
                print(f"[TRAIN] epoch = {epoch}, iteration = {'%04d'%i}, " + ', '.join([f"{k} = {strify(v)}" for k, v in results.items()]))
                i += 1; k += 1
            if i_epoch % n_epoch_valid == 0:
                if valid_info is not None: batch = dataset.get_batch(valid_info)
                else: batch = dataset.validation_batch(n_batch, restart = True)
                if batch is None: break
                results = func(('validation', i_epoch, 0, 0), batch)
                if not isinstance(results, dict): results = {'loss': results}
                print(f"[VALID] epoch = {epoch}, " + ', '.join([f"{k} = {strify(v)}" for k, v in results.items()]))
    return decorator

def test(dataset, n_batch=4, test_info=None):
    """
    Test by dataset. 
    Warning: Requires `Dataset` in package `micomputing`. 
    
    Note: Please use a function for loss computation (see example). It takes index and data batch. 
        index is a number, indicating the iteration of batch. It is `0` when `test_info` is given.

    Args:
        dataset (Dataset): The dataset structure.
        n_batch (int): Batch size for testing. Defaults to `4`.
        test_info (None or list): The manually specified batch to test.
    
    Examples::
        >>> @test(dataset)
        ... def loss(index, batch_data):
        ...     i_batch = index
        ...     model1.eval()
        ...     loss = loss_func(batch_data)
        ...     plt.plot(batch_data[0])
        ... return {'loss': loss}
    """
    def decorator(func):
        if test_info is not None:
            batch = dataset.get_batch(test_info)
            results = func(0, batch)
            if not isinstance(results, dict): results = {'loss': results}
            print("[TEST] " + ', '.join([f"{k} = {strify(v)}" for k, v in results.items()]))
        else:
            i = 1
            while True:
                batch = dataset.testing_batch(n_batch)
                if batch is None: break
                results = func(i, batch)
                if not isinstance(results, dict): results = {'loss': results}
                print(f"[TEST] ibatch = {i}, iteration = {'%04d'%i}, " + ', '.join([f"{k} = {strify(v)}" for k, v in results.items()]))
                i += 1
    return decorator
