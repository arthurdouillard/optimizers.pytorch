import torch


class AdaDelta(torch.optim.Optimizer):
    def __init__(self, params, weight_decay=0.0, epsilon=10e-08, momentum=0.9):
        defaults = dict(
            weight_decay=weight_decay,
            epsilon=epsilon, momentum=momentum
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                weight_decay = group['weight_decay']
                epsilon = group['epsilon']
                momentum = group['momentum']
                state = self.state[p]
                dp = p.grad

                if weight_decay != 0.:
                    dp.add(p.data, alpha=weight_decay)

                if 'G' in state:
                    state['G'] = momentum * state['G'] + (1 - momentum) * dp ** 2
                if 'G' not in state:
                    state['G'] = dp ** 2

                if 'D' not in state:
                    state['D'] = p.data ** 2

                factor = torch.sqrt((state['D'] + epsilon)/ (state['G'] + epsilon))

                p.data.add_(-factor * dp)
                if 'D' in state:
                    state['D'] = momentum * state['D'] + (1 - momentum) * p.data ** 2

