"""
Caching utilities for TorchRef modules.

Provides ``ParameterFingerprint`` for lightweight parameter-change detection
and ``CachedForwardMixin`` for automatic caching of ``forward()`` results
with invalidation on parameter mutation or backward propagation.

``CachedForwardMixin`` can be switched off globally via ``torchref.config.caching`` /
``TORCHREF_CACHING=0``, or for a block via :func:`no_caching`. ``ParameterFingerprint`` is
unaffected -- it is a standalone helper its users drive themselves.
"""

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


class CachedForwardMixin:
    """Mixin that caches ``forward()`` results with automatic invalidation.

    Overrides ``__call__`` to return a cached result while the module's parameters, buffers
    and call arguments are unchanged and no backward has propagated through the cached
    output. Invalidated by: any parameter/buffer ``(data_ptr, _version)`` change (so
    optimizer in-place updates and parameter replacement are both covered); any input
    tensor ``(data_ptr, _version)`` or non-tensor argument change; or a backward through
    the cached output, via a gradient hook that bumps a generation counter.

    The cached tensor **keeps its autograd graph**, so gradients flow on the first backward
    and the cache is invalidated after it -- a second backward on the same result needs
    ``retain_graph``. A caller that reads the result under ``no_grad`` and stores it will
    poison the cache with a detached tensor; call :meth:`reset_forward_cache` after.

    Fingerprints inline rather than via :class:`ParameterFingerprint`, which is a separate
    mechanism and also tracks ``numel``.

    Caching is on unless ``torchref.config.caching.value`` (env ``TORCHREF_CACHING``) is
    False, or the call is inside :func:`no_caching`; off, every call recomputes.
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

    def __getstate__(self):
        """Pickle and deepcopy carry no cached forward result.

        The cached output can hold an autograd-graph-attached tensor, which
        ``deepcopy`` refuses to walk, and the fingerprints are ``data_ptr``-based
        so they could never validate a copy anyway. The copy starts cold and
        recomputes on first call.
        """
        parent = getattr(super(), "__getstate__", None)
        state = dict(parent()) if parent is not None else dict(self.__dict__)
        for key in (
            "_fwd_cached_output",
            "_fwd_cached_state_fp",
            "_fwd_cached_input_fp",
            "_fwd_cache_gen",
            "_fwd_current_gen",
        ):
            state.pop(key, None)
        return state


@contextmanager
def no_caching():
    """Disable :class:`CachedForwardMixin` for the duration of the block.

    Restores the previous value of ``torchref.config.caching`` on exit, including when the
    body raises. Modules that already hold a cached result drop it on their first call
    inside the block, so nothing computed before entry survives to be served after it.

    Flips **process-global** state and is therefore not thread-safe: other threads see the
    change too, and nesting only restores correctly if the blocks are properly nested.

    Examples
    --------
    >>> with no_caching():
    ...     reference = model(hkl)  # doctest: +SKIP
    """
    previous = _caching_config.value
    _caching_config.value = False
    try:
        yield
    finally:
        _caching_config.value = previous
