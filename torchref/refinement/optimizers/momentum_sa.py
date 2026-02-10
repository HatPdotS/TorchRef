import torch
from torch import Tensor
from torch.optim.sgd import sgd
from torch.optim.optimizer import _use_grad_for_differentiable
from typing import Optional

'''
This is really just SGD with noisy gradients.
Temperature controls the magnitude of the noise relative to the gradient.

'''



class MomentumStochasticSA(torch.optim.Adam):
    """
    Adam-based SA where noise is scaled by the adaptive learning rate,
    giving automatic scale invariance across parameters.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 T_initial=1.0, T_final=0.01, total_steps=1000):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.temperatures = torch.logspace(
            torch.log10(torch.tensor(T_initial)),
            torch.log10(torch.tensor(T_final)),
            total_steps
        )
        self.current_step = 0
        self.total_steps = total_steps

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        T = self.temperatures[min(self.current_step, self.total_steps - 1)]
        self.current_step += 1

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Update biased moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Corrected moments
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Denominator (inverse "stiffness")
                denom = v_hat.sqrt() + eps

                # Standard Adam update
                p.addcdiv_(m_hat, denom, value=-lr)

                # Scale-invariant noise: soft directions get more noise
                noise = torch.randn_like(p) * (T / denom)
                p.add_(noise)

        return loss