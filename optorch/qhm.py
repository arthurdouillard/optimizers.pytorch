import torch


class QHM(torch.optim.Optimizer):
    """Quasi-Hyperbolic Momentum"""
    def __init__(self, params, lr=0.1, weight_decay=0.0, nu=0.7, beta=0.999):
        defaults = dict(
            lr=lr, weight_decay=weight_decay,
            nu=nu, beta=beta
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                weight_decay = group['weight_decay']
                lr = group['lr']
                nu, beta = group['nu'], group['beta']
                state = self.state[p]
                dp = p.grad

                if weight_decay != 0.:
                    dp.add(p.data, alpha=weight_decay)

                if 'buffer' in state:
                    buf = beta * state['buffer'] + (1 - beta) * dp
                if 'buffer' not in state:
                    state['buffer'] = dp
                    buf = 0

                factor = (1 - nu) * dp + nu * buf

                p.data.add_(-lr, factor)
