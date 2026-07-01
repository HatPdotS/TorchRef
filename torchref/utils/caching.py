"""
Caching utilities for TorchRef modules.

Provides ``ParameterFingerprint`` for lightweight parameter-change detection
and ``CachedForwardMixin`` for automatic caching of ``forward()`` results
with invalidation on parameter mutation or backward propagation.
"""

import torch


class ParameterFingerprint:
    """Lightweight fingerprint for detecting parameter changes.

    Captures (data_ptr, _version, numel) per tensor. Comparison is O(n_params)
    integer comparisons — much cheaper than SHA-1 hashing.
    """

    __slots__ = ("_entries",)

    def __init__(self, params=()):
        self._entries = tuple(
            (t.data_ptr(), t._version, t.numel()) for t in params
        )

    def matches(self, params) -> bool:
        """Return True if *params* have the same fingerprint."""
        other = tuple(
            (t.data_ptr(), t._version, t.numel()) for t in params
        )
        return self._entries == other

    def __bool__(self):
        """True iff the fingerprint captured at least one tensor.

        Note this reports whether the fingerprint is *non-empty*, not
        whether it *matches* anything — use :meth:`matches` for comparison.
        """
        return len(self._entries) > 0


class CachedForwardMixin:
    """Mixin that caches ``forward()`` results with automatic invalidation.

    Overrides ``__call__`` to return a cached result when the module's
    parameters, buffers, and call arguments have not changed since the
    last invocation — and no backward pass has propagated through the
    cached output.

    Cache invalidation triggers:

    * Any parameter or buffer ``data_ptr`` or ``_version`` change
      (covers optimizer in-place updates and mask/parameter replacement).
    * Input tensor ``data_ptr`` or ``_version`` change, or non-tensor
      argument value change.
    * A backward pass through the cached output (increments generation
      counter via a gradient hook).

    The cached tensor retains its autograd graph — gradients flow correctly
    on the first backward pass, after which the cache is invalidated.

    Notes
    -----
    This mixin does **not** use :class:`ParameterFingerprint`. The two
    change-detection mechanisms are independent: ``ParameterFingerprint``
    captures a ``(data_ptr, _version, numel)`` triple per tensor, whereas
    this mixin fingerprints inline via ``_fingerprint_state`` /
    ``_fingerprint_inputs`` using a ``(data_ptr, _version)`` pair (no
    ``numel``).
    """

    # ---- internal helpers ------------------------------------------------

    def _fingerprint_state(self):
        """Fingerprint all parameters and buffers by ``(data_ptr, _version)``."""
        entries = []
        for t in self.parameters():
            entries.append((t.data_ptr(), t._version))
        for t in self.buffers():
            entries.append((t.data_ptr(), t._version))
        return tuple(entries)

    @staticmethod
    def _fingerprint_inputs(args, kwargs):
        """Fingerprint call arguments (tensor ptr/version, else by value)."""
        entries = []
        for a in args:
            if isinstance(a, torch.Tensor):
                entries.append((a.data_ptr(), a._version))
            else:
                entries.append(a)
        for k in sorted(kwargs):
            v = kwargs[k]
            if isinstance(v, torch.Tensor):
                entries.append((k, v.data_ptr(), v._version))
            else:
                entries.append((k, v))
        return tuple(entries)

    # ---- public API ------------------------------------------------------

    def __call__(self, *args, recalc=False, **kwargs):
        """Return cached ``forward()`` result, or recompute on cache miss.

        Parameters
        ----------
        *args
            Positional arguments forwarded to ``forward()`` (and fingerprinted
            for cache validity).
        recalc : bool, optional
            If True, invalidate the cache and force recomputation.
            Not forwarded to ``forward()``.
        **kwargs
            Keyword arguments forwarded to ``forward()`` (and fingerprinted
            for cache validity).

        Returns
        -------
        object
            The cached ``forward()`` result on a cache hit, otherwise the
            freshly computed result.
        """
        if recalc:
            self.reset_forward_cache()

        cached = getattr(self, "_fwd_cached_output", None)
        if cached is not None:
            state_fp = self._fingerprint_state()
            input_fp = self._fingerprint_inputs(args, kwargs)
            gen = getattr(self, "_fwd_current_gen", 0)
            if (
                state_fp == self._fwd_cached_state_fp
                and input_fp == self._fwd_cached_input_fp
                and gen == self._fwd_cache_gen
            ):
                return cached

        # Cache miss — recompute
        result = self.forward(*args, **kwargs)

        # Register backward hook to invalidate cache after gradient consumption
        if isinstance(result, torch.Tensor) and result.grad_fn is not None:
            def _bump_gen(grad, ref=self):
                ref._fwd_current_gen = getattr(ref, "_fwd_current_gen", 0) + 1
            result.register_hook(_bump_gen)

        # Store cache state
        self._fwd_cached_output = result
        self._fwd_cached_state_fp = self._fingerprint_state()
        self._fwd_cached_input_fp = self._fingerprint_inputs(args, kwargs)
        if not hasattr(self, "_fwd_current_gen"):
            self._fwd_current_gen = 0
        self._fwd_cache_gen = self._fwd_current_gen

        return result

    def reset_forward_cache(self):
        """Manually invalidate the forward cache."""
        self._fwd_cached_output = None
        self._fwd_cached_state_fp = None
        self._fwd_cached_input_fp = None
        self._fwd_cache_gen = 0
        self._fwd_current_gen = 0
