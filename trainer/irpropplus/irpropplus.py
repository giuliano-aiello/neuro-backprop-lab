import torch
from torch.optim.optimizer import Optimizer

class IRpropPlus(Optimizer):
    def __init__(self, params, lr=0.012, etaminus=0.5, etaplus=1.2, delta_min=1e-6, delta_max=50):
        defaults = dict(lr=lr, etaminus=etaminus, etaplus=etaplus, delta_min=delta_min, delta_max=delta_max)
        super(IRpropPlus, self).__init__(params, defaults)

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
                    state['prev_error'] = torch.zeros_like(parameter.data)
                    state['prev_update'] = torch.zeros_like(parameter.data)

                step_size = state['step_size']
                prev_grad = state['prev_grad']
                prev_error = state['prev_error']
                prev_update = state['prev_update']

                grad_sign = grad * prev_grad
                stepsize_increase = grad_sign > 0
                stepsize_decrease = grad_sign < 0

                error = grad.abs()

                step_size[stepsize_increase] = torch.min(step_size[stepsize_increase] * eta_plus, torch.tensor(delta_max, device=parameter.device))

                step_size[stepsize_decrease] = torch.where(error[stepsize_decrease] > prev_error[stepsize_decrease],
                                                       -step_size[stepsize_decrease],
                                                       torch.zeros_like(step_size[stepsize_decrease]))
                step_size[stepsize_decrease] = torch.max(step_size[stepsize_decrease] * eta_minus, torch.tensor(delta_min, device=parameter.device))

                update = torch.zeros_like(parameter.data)
                update[grad_sign >= 0] = -torch.sign(grad[grad_sign >= 0]) * step_size[grad_sign >= 0]
                update[grad_sign < 0] = -prev_update[grad_sign < 0]
                grad[grad_sign < 0] = 0

                parameter.data += update

                state['prev_grad'].copy_(grad)
                state['prev_error'].copy_(error)
                state['prev_update'].copy_(update)
