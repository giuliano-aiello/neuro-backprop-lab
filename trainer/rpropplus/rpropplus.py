import torch
from torch.optim.optimizer import Optimizer

class RpropPlus(Optimizer):
    def __init__(self, params, lr=0.012, etaminus=0.5, etaplus=1.2, delta_min=1e-6, delta_max=50):
        defaults = dict(lr=lr, etaminus=etaminus, etaplus=etaplus, delta_min=delta_min, delta_max=delta_max)
        super(RpropPlus, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            eta_plus = group['etaplus']
            eta_minus = group['etaminus']
            delta_min = group['delta_min']
            delta_max = group['delta_max']

            for parameter in group['params']:
                if parameter.grad is None:
                    continue

                grad = parameter.grad.detach()
                state = self.state[parameter]

                if 'step_size' not in state:
                    state['step_size'] = torch.full_like(parameter.data, group['lr'])
                    state['prev_grad'] = torch.zeros_like(parameter.data)
                    state['prev_update'] = torch.zeros_like(parameter.data)

                step_size = state['step_size']
                prev_grad = state['prev_grad']
                prev_update = state['prev_update']

                grad_sign = grad * prev_grad

                increase_mask = grad_sign > 0
                decrease_mask = grad_sign < 0

                step_size[increase_mask] = torch.min(step_size[increase_mask] * eta_plus, torch.tensor(delta_max, device=parameter.device))         # device comunica su quale dispositivo (CPU o GPU) avviene il calolo
                step_size[decrease_mask] = torch.max(step_size[decrease_mask] * eta_minus, torch.tensor(delta_min, device=parameter.device))

                update = torch.zeros_like(parameter.data)
                update[increase_mask | (grad_sign == 0)] = -torch.sign(grad[increase_mask | (grad_sign == 0)]) * step_size[increase_mask | (grad_sign == 0)]
                update[decrease_mask] = -prev_update[decrease_mask]
                grad[decrease_mask] = 0

                parameter.data += update

                state['prev_grad'].copy_(grad)
                state['prev_update'].copy_(update)
