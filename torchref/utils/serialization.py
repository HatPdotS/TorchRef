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
        A JSON-serializable equivalent. Note the shape asymmetry: a one-element tensor
        collapses to a **scalar** via ``.item()`` while anything longer becomes a list, so a
        shape-``(1,)`` tensor does not round-trip to a list.
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
