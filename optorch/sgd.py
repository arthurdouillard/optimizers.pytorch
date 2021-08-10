import torch


class SGD(torch.optim.Optimizer):
    """Classic SGD optimizer"""
    def __init__(self, params, lr, weight_decay=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                weight_decay = group['weight_decay']
                lr = group['lr']
                dp = p.grad

                if weight_decay != 0.:
                    # L = l + 0.5 * ||w||^2
                    # G = g + w
                    dp.add(p.data, alpha=weight_decay)

                p.data.add_(-lr, dp)
