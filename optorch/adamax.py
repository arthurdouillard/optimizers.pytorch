import torch


class AdaMax(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, weight_decay=0.0, beta1=0.9, beta2=0.999, epsilon=10e-08):
        defaults = dict(
            lr=lr, weight_decay=weight_decay,
            beta1=beta1, beta2=beta2,
            epsilon=epsilon
        )
        self.t = 1
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                weight_decay = group['weight_decay']
                lr = group['lr']
                beta1, beta2 = group['beta1'], group['beta2']
                epsilon = group['epsilon']
                state = self.state[p]
                dp = p.grad

                if weight_decay != 0.:
                    dp.add(p.data, alpha=weight_decay)

                if 'm' in state:
                    m = beta1 * state['m'] + (1 - beta1) * dp
                    v = beta2 * state['v'] + (1 - beta2) * dp.abs()
                    u = torch.maximum(
                        beta2 * state['v'], dp.abs()
                    )
                    state['m'] = m
                    state['v'] = v
                if 'm' not in state:
                    state['m'] = torch.clone(dp).detach()
                    state['v'] = torch.clone(dp).detach().abs()
                    u = state['v']

                mhat = state['m'] / (1 - beta1 ** self.t)
                factor = mhat / u

                p.data.add_(-lr, factor)
        self.t += 1

