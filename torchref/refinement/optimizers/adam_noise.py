"""
Adam optimizer with adaptive noise injection for regularization.

This optimizer extends Adam with scale-invariant Gaussian noise injection
to prevent overfitting during crystallographic refinement.
"""

import torch
from torch.optim import Adam


class AdamWithAdaptiveNoise(Adam):
    """
    Drop-in replacement for torch.optim.Adam with adaptive, scale-invariant noise injection.

    Injects Gaussian noise into gradients scaled by the overfitting ratio between
    training and test NLL to prevent overfitting.

    Parameters
    ----------
    params : iterable
        Model parameters to optimize.
    lr : float, optional
        Learning rate. Default is 1e-3.
    alpha : float, optional
        Scaling factor for how much noise to inject per unit overfitting ratio.
        Default is 0.1.
    eps : float, optional
        Small constant for numerical stability. Default is 1e-8.
    update_weight : float, optional
        Weight for exponential moving average of noise scale. Default is 0.05.
    **kwargs
        Additional arguments passed to Adam optimizer.

    Attributes
    ----------
    alpha : float
        Noise scaling factor.
    eps : float
        Numerical stability constant.
    noise_scale : float
        Current noise scale (dynamically updated).
    update_weight : float
        EMA weight for noise scale updates.
    """

    def __init__(
        self, params, lr=1e-3, alpha=0.1, eps=1e-8, update_weight=0.05, **kwargs
    ):
        """
        Initialize AdamWithAdaptiveNoise.

        Parameters
        ----------
        params : iterable
            Model parameters to optimize.
        lr : float, optional
            Learning rate. Default is 1e-3.
        alpha : float, optional
            Scaling factor for how much noise to inject per unit overfitting ratio.
            Default is 0.1.
        eps : float, optional
            Small constant for numerical stability. Default is 1e-8.
        update_weight : float, optional
            Weight for exponential moving average of noise scale. Default is 0.05.
        **kwargs
            Additional arguments passed to Adam optimizer.
        """
        super().__init__(params, lr=lr, **kwargs)
        self.alpha = alpha
        self.eps = eps
        self.noise_scale = 0.0  # dynamically updated
        self.update_weight = update_weight

    @torch.no_grad()
    def inject_noise(self):
        """
        Inject scale-invariant Gaussian noise into gradients.

        The noise standard deviation is proportional to the gradient and parameter
        norms, scaled by the current noise_scale and alpha.
        """
        if self.noise_scale <= 0:
            return
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                gradnorm = torch.norm(grad)
                paramnorm = torch.norm(p)
                rms = paramnorm * 0.01 + gradnorm
                noise_std = self.noise_scale * self.alpha * rms
                grad.add_(torch.randn_like(grad) * noise_std)

    def step(self):
        """
        Perform a single optimization step with optional noise injection.

        Injects noise into gradients before the Adam update if noise_scale > 0.
        """
        # Inject noise before the Adam update
        self.inject_noise()
        super().step()

    def update_noise_scale(self, train_nll, test_nll):
        """
        Update the noise scale based on the ratio of test to training NLL.

        If ratio > 1, the model is overfitting and noise is increased.

        Parameters
        ----------
        train_nll : torch.Tensor
            Training set negative log-likelihood.
        test_nll : torch.Tensor
            Test set negative log-likelihood.
        """
        ratio = torch.log(torch.clamp(train_nll, min=1e-4)) - torch.log(
            torch.clamp(test_nll, min=1e-4)
        )
        ratio = torch.clamp(ratio, min=0.0, max=0.1)  # only consider overfitting
        self.noise_scale = (
            self.update_weight * ratio.item()
            + (1 - self.update_weight) * self.noise_scale
        )
