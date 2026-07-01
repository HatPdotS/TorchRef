"""
Adam-based simulated annealing optimizer with adaptive noise injection.

Implements Adam with added Gaussian noise on each step. The noise magnitude is
controlled by a temperature that is annealed on a logarithmic schedule from
``T_initial`` to ``T_final`` and scaled by Adam's adaptive denominator, giving
scale-invariant exploration across parameters.
"""

import torch
from torch import Tensor
from torch.optim.sgd import sgd
from torch.optim.optimizer import _use_grad_for_differentiable
from typing import Optional



class MomentumStochasticSA(torch.optim.Adam):
    """
    Adam-based SA where noise is scaled by the adaptive learning rate,
    giving automatic scale invariance across parameters.

    At each step Gaussian noise scaled by ``T / denom`` is added after the
    Adam update, where ``denom`` is Adam's adaptive denominator (so soft,
    low-curvature directions receive more noise) and ``T`` is the current
    temperature. Temperature is annealed from ``T_initial`` to ``T_final``
    over ``total_steps`` on a logarithmic schedule (``torch.logspace``).

    Parameters
    ----------
    params : iterable
        Parameters to optimize, passed through to ``torch.optim.Adam``.
    lr : float, optional
        Learning rate. Default is 1e-3.
    betas : tuple of float, optional
        Adam moment decay coefficients. Default is (0.9, 0.999).
    eps : float, optional
        Term added to the denominator for numerical stability.
        Default is 1e-8.
    T_initial : float, optional
        Starting temperature of the logarithmic annealing schedule.
        Default is 1.0.
    T_final : float, optional
        Ending temperature of the logarithmic annealing schedule.
        Default is 0.01.
    total_steps : int, optional
        Number of steps over which temperature is annealed. After this many
        steps the temperature is held at ``T_final``. Default is 1000.
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
        """
        Perform a single Adam update followed by scale-invariant noise.

        Reads the temperature for the current step, applies the standard
        Adam parameter update, then adds Gaussian noise scaled by
        ``T / denom`` so that low-curvature directions are explored more.

        Parameters
        ----------
        closure : callable, optional
            Closure that re-evaluates the model and returns the loss. If
            provided, it is called under ``torch.enable_grad`` and its return
            value is passed back.

        Returns
        -------
        loss : torch.Tensor or None
            The loss returned by ``closure``, or None if no closure was given.
        """
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