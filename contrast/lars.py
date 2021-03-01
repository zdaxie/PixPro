import torch
from torch.optim.optimizer import Optimizer

__all__ = ['LARS']


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """Splits param group into weight_decay / non-weight decay.
       Tweaked from https://bit.ly/3dzyqod
    :param model: the torch.nn model
    :param weight_decay: weight decay term
    :param skip_list: extra modules (besides BN/bias) to skip
    :returns: split param group into weight_decay/not-weight decay
    :rtype: list(dict)
    """
    # if weight_decay == 0:
    #     return model.parameters()

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name in skip_list:
            # print(name)
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay':  0, 'ignore': True},
        {'params': decay, 'weight_decay': weight_decay, 'ignore': False}]

class LARS(Optimizer):
    """Implements 'LARS (Layer-wise Adaptive Rate Scaling)'__ as Optimizer a
    :class:`~torch.optim.Optimizer` wrapper.

    __ : https://arxiv.org/abs/1708.03888

    Wraps an arbitrary optimizer like :class:`torch.optim.SGD` to use LARS. If
    you want to the same performance obtained with small-batch training when
    you use large-batch training, LARS will be helpful::

    Args:
        optimizer (Optimizer):
            optimizer to wrap
        eps (float, optional):
            epsilon to help with numerical stability while calculating the
            adaptive learning rate
        trust_coef (float, optional):
            trust coefficient for calculating the adaptive learning rate

    Example::
        base_optimizer = optim.SGD(model.parameters(), lr=0.1)
        optimizer = LARS(optimizer=base_optimizer)

        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()

    """

    def __init__(self, optimizer, eps=1e-8, trust_coef=0.001):
        if eps < 0.0:
            raise ValueError('invalid epsilon value: , %f' % eps)

        if trust_coef < 0.0:
            raise ValueError("invalid trust coefficient: %f" % trust_coef)

        self.optim = optimizer
        self.eps = eps
        self.trust_coef = trust_coef

    def __getstate__(self):
        lars_dict = {}
        lars_dict['eps'] = self.eps
        lars_dict['trust_coef'] = self.trust_coef
        return (self.optim, lars_dict)

    def __setstate__(self, state):
        self.optim, lars_dict = state
        self.eps = lars_dict['eps']
        self.trust_coef = lars_dict['trust_coef']

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.optim)

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def state(self):
        return self.optim.state

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def apply_adaptive_lrs(self):
        with torch.no_grad():
            for group in self.optim.param_groups:
                weight_decay = group['weight_decay']
                ignore = group.get('ignore', None)  # NOTE: this is set by add_weight_decay

                for p in group['params']:
                    if p.grad is None:
                        continue

                    # Add weight decay before computing adaptive LR
                    # Seems to be pretty important in SIMclr style models.
                    if weight_decay > 0:
                        p.grad = p.grad.add(p, alpha=weight_decay)

                    # Ignore bias / bn terms for LARS update
                    if ignore is not None and not ignore:
                        # compute the parameter and gradient norms
                        param_norm = p.norm()
                        grad_norm = p.grad.norm()

                        # compute our adaptive learning rate
                        adaptive_lr = 1.0
                        if param_norm > 0 and grad_norm > 0:
                            adaptive_lr = self.trust_coef * param_norm / (grad_norm + self.eps)

                        # print("applying {} lr scaling to param of shape {}".format(adaptive_lr, p.shape))
                        p.grad = p.grad.mul(adaptive_lr)

    def step(self, *args, **kwargs):
        self.apply_adaptive_lrs()

        # Zero out weight decay as we do it in LARS
        weight_decay_orig = [group['weight_decay'] for group in self.optim.param_groups]
        for group in self.optim.param_groups:
            group['weight_decay'] = 0

        loss = self.optim.step(*args, **kwargs)  # Normal optimizer

        # Restore weight decay
        for group, wo in zip(self.optim.param_groups, weight_decay_orig):
            group['weight_decay'] = wo

        return loss