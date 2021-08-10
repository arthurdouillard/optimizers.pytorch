import torch


class SGDW(torch.optim.Optimizer):
    """SGD with Momentum and Decoupled weight decay"""
    def __init__(self, params, lr, weight_decay=0.0, momentum=0.0):
        defaults = dict(
            lr=lr, weight_decay=weight_decay,
            momentum=momentum
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
                momentum = group['momentum']
                state = self.state[p]
                dp = p.grad

                if momentum != 0.:
                    if 'momentum_buffer' in state:
                        dp.add(state['momentum_buffer'], alpha=momentum)
                        state['momentum_buffer'].mul_(momentum).add_(dp)
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.clone(dp).detach()

                    # g' = (momentum * buffer + g) - lambda * theta
                    dp = state['momentum_buffer']
                    dp = dp.add_(p.data, alpha=-weight_decay)

                p.data.add_(-lr, dp)
