import torch


class SGDM(torch.optim.Optimizer):
    """SGD with Momentum"""
    def __init__(self, params, lr=0.1, weight_decay=0.0, momentum=0.9):
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

                if weight_decay != 0.:
                    dp.add(p.data, alpha=weight_decay)

                if momentum != 0.:
                    if 'momentum_buffer' in state:
                        dp.add(state['momentum_buffer'], alpha=momentum)
                        state['momentum_buffer'].mul_(momentum).add_(dp)
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.clone(dp).detach()

                    # g' = momentum * buffer + g
                    dp = state['momentum_buffer']

                p.data.add_(-lr, dp)
