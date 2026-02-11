"""
DeviceMovementMixin — eliminates cuda()/cpu() boilerplate.

Classes that define a custom ``to(device=..., dtype=...)`` method can inherit
this mixin to get standard ``cuda()`` and ``cpu()`` methods for free.

Usage::

    class MyModule(DeviceMovementMixin, nn.Module):
        def to(self, device=None, dtype=None):
            # custom logic
            ...
            return self

        # cuda() and cpu() are now provided automatically
"""

from __future__ import annotations


class DeviceMovementMixin:
    """Mixin providing standard ``cuda()`` and ``cpu()`` that delegate to ``to()``."""

    def cuda(self, device=None):
        """Move all tensors to a CUDA device.

        Parameters
        ----------
        device : int or str or torch.device, optional
            CUDA device index (e.g. 0) or full device string.
            Defaults to ``"cuda"``.

        Returns
        -------
        self
        """
        if device is None:
            device = "cuda"
        elif isinstance(device, int):
            device = f"cuda:{device}"
        return self.to(device=device)

    def cpu(self):
        """Move all tensors to CPU.

        Returns
        -------
        self
        """
        return self.to(device="cpu")
