"""
Caching utilities for TorchRef modules.

Provides ``ParameterFingerprint`` for lightweight parameter-change detection
and ``CachedForwardMixin`` for automatic caching of ``forward()`` results
with invalidation on parameter mutation or backward propagation.

``CachedForwardMixin`` can be switched off globally via ``torchref.config.caching`` /
``TORCHREF_CACHING=0``, or for a block via :func:`no_caching`. ``ParameterFingerprint`` is
unaffected -- it is a standalone helper its users drive themselves.
"""

import weakref
from contextlib import contextmanager

import torch

from torchref.config import caching as _caching_config
from torchref.config import get_caching_enabled


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
        """Non-empty, **not** "matches" -- use :meth:`matches` to compare."""
        return len(self._entries) > 0


class _TensorKey:
    """Cache key for one tensor: the tensor itself, held weakly, plus its storage state.

    Equal only to a key for the *same live tensor object* with the same ``data_ptr``,
    ``_version`` and ``requires_grad``. ``data_ptr`` alone cannot identify a tensor: the
    allocator hands a freed tensor's address to the next allocation, whose ``_version``
    also starts at 0, so a new ``hkl`` would otherwise be served the old one's result.
    """

    __slots__ = ("_ref", "_state")

    def __init__(self, t: torch.Tensor):
        self._ref = weakref.ref(t)
        self._state = (t.data_ptr(), t._version, t.requires_grad)

    def __eq__(self, other) -> bool:
        if not isinstance(other, _TensorKey):
            return NotImplemented
        t = self._ref()
        return t is not None and t is other._ref() and self._state == other._state

    __hash__ = None


class CachedForwardMixin:
    """Mixin that caches ``forward()`` results with automatic invalidation.

    Overrides ``__call__`` to return a cached result while the module's parameters, buffers
    and call arguments are unchanged and no backward has propagated through the cached
    output. Tensors -- parameters, buffers and inputs alike -- are matched by identity (the
    same live object, held weakly) and ``(data_ptr, _version, requires_grad)``, so a new
    tensor never matches, even one allocated at a freed tensor's address. Invalidated by:
    an optimizer in-place update, parameter replacement or freezing; any of those on an
    input tensor, a different input tensor, or a non-tensor argument change; or a
    backward through the cached output, via a gradient hook that bumps a generation
    counter. A write through ``.data`` (``p.data.copy_(x)``) leaves ``data_ptr`` and
    ``_version`` as they were, so it is served stale: write under ``torch.no_grad()``
    without ``.data``, or call :meth:`reset_forward_cache` after.

    The cached tensor **keeps its autograd graph**, so gradients flow on the first backward
    and the cache is invalidated after it -- a second backward on the same result needs
    ``retain_graph``. A result computed with grad disabled has no graph, so it is served
    only while grad stays disabled; a grad-enabled call recomputes it.

    Fingerprints inline rather than via :class:`ParameterFingerprint`, which is a separate
    mechanism and also tracks ``numel``.

    Caching is on unless ``torchref.config.caching.value`` (env ``TORCHREF_CACHING``) is
    False, or the call is inside :func:`no_caching`; off, every call recomputes.
    """

    # ---- internal helpers ------------------------------------------------

    def _fingerprint_state(self):
        """Key every parameter and buffer by identity and storage state."""
        entries = [_TensorKey(t) for t in self.parameters()]
        entries.extend(_TensorKey(t) for t in self.buffers())
        return tuple(entries)

    @staticmethod
    def _fingerprint_inputs(args, kwargs):
        """Key call arguments: tensors as in the state key, the rest by value."""
        entries = []
        for a in args:
            if isinstance(a, torch.Tensor):
                entries.append(_TensorKey(a))
            else:
                entries.append(a)
        for k in sorted(kwargs):
            v = kwargs[k]
            if isinstance(v, torch.Tensor):
                entries.append((k, _TensorKey(v)))
            else:
                entries.append((k, v))
        return tuple(entries)

    # ---- public API ------------------------------------------------------

    def __call__(self, *args, recalc=False, **kwargs):
        """Return the cached ``forward()`` result, or recompute on a cache miss.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to ``forward()`` and fingerprinted for cache validity.
        recalc : bool, optional
            Invalidate the cache and recompute. Consumed here, not forwarded.

        Notes
        -----
        With caching disabled (``torchref.config.caching``) this is a plain call to
        ``forward()``; ``recalc`` is still consumed rather than forwarded.
        """
        if recalc:
            self.reset_forward_cache()

        if not get_caching_enabled():
            # Drop anything cached before the flag flipped, so re-enabling cannot serve a
            # result computed under parameters that have since moved on. Guarded rather
            # than unconditional: this is the hot path and the cache is usually empty.
            if getattr(self, "_fwd_cached_output", None) is not None:
                self.reset_forward_cache()
            return self.forward(*args, **kwargs)

        cached = getattr(self, "_fwd_cached_output", None)
        if cached is not None:
            state_fp = self._fingerprint_state()
            input_fp = self._fingerprint_inputs(args, kwargs)
            gen = getattr(self, "_fwd_current_gen", 0)
            # One-sided: a result computed without grad has no graph to give a grad-mode
            # call, while a graph-carrying result is still right under ``no_grad``.
            if (
                state_fp == self._fwd_cached_state_fp
                and input_fp == self._fwd_cached_input_fp
                and gen == self._fwd_cache_gen
                and (self._fwd_cached_with_grad or not torch.is_grad_enabled())
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
        self._fwd_cached_with_grad = torch.is_grad_enabled()
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


@contextmanager
def no_caching():
    """Disable :class:`CachedForwardMixin` for the duration of the block.

    Restores the previous value of ``torchref.config.caching`` on exit, including when the
    body raises. Modules that already hold a cached result drop it on their first call
    inside the block, so nothing computed before entry survives to be served after it.

    Flips **process-global** state and is therefore not thread-safe: other threads see the
    change too, and nesting only restores correctly if the blocks are properly nested.
    """
    previous = _caching_config.value
    _caching_config.value = False
    try:
        yield
    finally:
        _caching_config.value = previous
