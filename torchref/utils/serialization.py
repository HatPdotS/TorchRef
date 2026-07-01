"""JSON serialization helpers for torch tensors and numpy arrays."""

import torch


def convert_to_serializable(obj):
    """Convert tensors and numpy arrays to JSON-serializable types.

    Recursively walks dicts, lists, and tuples, converting ``torch.Tensor``,
    ``numpy.ndarray``, and numpy scalar types to plain Python objects that
    ``json.dump`` can handle.

    Parameters
    ----------
    obj : object
        Arbitrary Python object (tensor, array, dict, list, scalar, ...).

    Returns
    -------
    object
        A JSON-serializable equivalent.

    Notes
    -----
    Tensor conversion depends on element count: a tensor with more than one
    element is converted via ``.tolist()``, while a tensor with exactly one
    element is converted to a Python scalar via ``.item()``. As a result a
    shape-``(1,)`` tensor collapses to a scalar (not a one-element list),
    whereas a shape-``(2,)`` tensor becomes a list — an asymmetry that can
    surprise round-trips.
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.numel() > 1 else obj.item()
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(convert_to_serializable(v) for v in obj)
    return obj
