import torch


class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, weight_decay=0.0, epsilon=10e-08):
        defaults = dict(
            lr=lr, weight_decay=weight_decay,
            epsilon=epsilon
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
                epsilon = group['epsilon']
                state = self.state[p]
                dp = p.grad

                if weight_decay != 0.:
                    dp.add(p.data, alpha=weight_decay)

                if 'G' in state:
                    state['G'] = state['G'] + dp ** 2
                if 'G' not in state:
                    state['G'] = dp ** 2

                factor = dp / torch.sqrt(state['G'] + epsilon)

                p.data.add_(-lr, factor)
