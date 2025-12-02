"""
Debug utilities for multicopy_refinement modules.

Provides debugging and introspection functionality for all module classes.
"""

import torch
import traceback
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd


class DebugMixin:
    """
    Mixin class that adds debugging capabilities to modules.

    When an error occurs, call print_debug_summary() to get a comprehensive
    overview of the module's state including:

    - All attributes and their types
    - Tensor shapes, dtypes, and devices
    - DataFrame/array shapes
    - Other object information
    """
    
    def print_debug_summary(self, title: str = None, file=sys.stderr):
        """
        Print a comprehensive debug summary of this module's state.

        Parameters
        ----------
        title : str, optional
            Title for the summary.
        file : file-like, default sys.stderr
            File to write output to.
        """
        if title is None:
            title = f"{self.__class__.__name__} Debug Summary"
        
        print("\n" + "=" * 80, file=file)
        print(f"  {title}", file=file)
        print("=" * 80, file=file)
        
        # Get all attributes
        attrs = {}
        for attr_name in dir(self):
            # Skip private/magic methods and callables (except modules)
            if attr_name.startswith('_'):
                continue
            
            try:
                attr_value = getattr(self, attr_name)
                
                # Skip methods unless they're submodules
                if callable(attr_value) and not isinstance(attr_value, torch.nn.Module):
                    continue
                
                attrs[attr_name] = attr_value
            except Exception as e:
                attrs[attr_name] = f"<Error accessing: {e}>"
        
        # Categorize and print attributes
        tensors = {}
        modules = {}
        dataframes = {}
        arrays = {}
        others = {}
        
        for name, value in attrs.items():
            if isinstance(value, torch.Tensor):
                tensors[name] = value
            elif isinstance(value, torch.nn.Module):
                modules[name] = value
            elif isinstance(value, pd.DataFrame):
                dataframes[name] = value
            elif isinstance(value, (np.ndarray, list, tuple)):
                arrays[name] = value
            else:
                others[name] = value
        
        # Print tensors with detailed info
        if tensors:
            print("\n📊 TENSORS:", file=file)
            print("-" * 80, file=file)
            for name, tensor in sorted(tensors.items()):
                try:
                    device_str = str(tensor.device)
                    dtype_str = str(tensor.dtype).replace('torch.', '')
                    shape_str = 'x'.join(map(str, tensor.shape)) if tensor.shape else 'scalar'
                    mem_mb = tensor.element_size() * tensor.numel() / (1024 * 1024)

                    # Get value info for small tensors
                    value_info = ""
                    if tensor.numel() <= 5:
                        value_info = f" = {tensor.detach().cpu().numpy()}"
                    elif tensor.numel() > 0:
                        val_min = tensor.min().item()
                        val_max = tensor.max().item()
                        val_mean = tensor.mean().item() if tensor.is_floating_point() else "N/A"
                        value_info = f" | range: [{val_min:.3g}, {val_max:.3g}], mean: {val_mean}"
                    
                    print(f"  {name:30s} : {dtype_str:12s} | shape: {shape_str:20s} | "
                          f"device: {device_str:10s} | mem: {mem_mb:.2f} MB{value_info}", 
                          file=file)
                except Exception as e:
                    print(f"  {name:30s} : <Error: {e}>", file=file)
        
        # Print submodules
        if modules:
            print("\n🔧 SUBMODULES:", file=file)
            print("-" * 80, file=file)
            for name, module in sorted(modules.items()):
                try:
                    module_type = type(module).__name__
                    
                    # Count parameters if available
                    try:
                        n_params = sum(p.numel() for p in module.parameters())
                        param_info = f" | params: {n_params:,}"
                    except:
                        param_info = ""
                    
                    # Check device
                    try:
                        device = next(module.parameters()).device
                        device_info = f" | device: {device}"
                    except:
                        device_info = ""
                    
                    print(f"  {name:30s} : {module_type}{param_info}{device_info}", file=file)
                except Exception as e:
                    print(f"  {name:30s} : <Error: {e}>", file=file)
        
        # Print DataFrames
        if dataframes:
            print("\n📋 DATAFRAMES:", file=file)
            print("-" * 80, file=file)
            for name, df in sorted(dataframes.items()):
                try:
                    shape_str = f"{df.shape[0]} rows x {df.shape[1]} cols"
                    cols_str = ', '.join(df.columns[:5].tolist())
                    if len(df.columns) > 5:
                        cols_str += f", ... ({len(df.columns)} total)"
                    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                    
                    print(f"  {name:30s} : {shape_str:20s} | mem: {mem_mb:.2f} MB", file=file)
                    print(f"  {' '*30}   columns: {cols_str}", file=file)
                except Exception as e:
                    print(f"  {name:30s} : <Error: {e}>", file=file)
        
        # Print arrays
        if arrays:
            print("\n🔢 ARRAYS/LISTS:", file=file)
            print("-" * 80, file=file)
            for name, arr in sorted(arrays.items()):
                try:
                    if isinstance(arr, np.ndarray):
                        shape_str = 'x'.join(map(str, arr.shape))
                        dtype_str = str(arr.dtype)
                        mem_mb = arr.nbytes / (1024 * 1024)
                        print(f"  {name:30s} : numpy.ndarray | dtype: {dtype_str:12s} | "
                              f"shape: {shape_str:20s} | mem: {mem_mb:.2f} MB", file=file)
                    elif isinstance(arr, (list, tuple)):
                        type_name = type(arr).__name__
                        len_str = f"len={len(arr)}"
                        if len(arr) > 0:
                            elem_type = type(arr[0]).__name__
                            len_str += f", first elem type: {elem_type}"
                        print(f"  {name:30s} : {type_name} | {len_str}", file=file)
                except Exception as e:
                    print(f"  {name:30s} : <Error: {e}>", file=file)
        
        # Print other attributes
        if others:
            print("\n📝 OTHER ATTRIBUTES:", file=file)
            print("-" * 80, file=file)
            for name, value in sorted(others.items()):
                try:
                    type_name = type(value).__name__
                    value_repr = repr(value)
                    
                    # Truncate long representations
                    if len(value_repr) > 60:
                        value_repr = value_repr[:57] + "..."
                    
                    print(f"  {name:30s} : {type_name:20s} = {value_repr}", file=file)
                except Exception as e:
                    print(f"  {name:30s} : <Error: {e}>", file=file)
        
        print("\n" + "=" * 80, file=file)
        print(file=file)
    
    def debug_on_error(self, error: Exception, context: str = "", recursive: bool = True):
        """
        Print debug summary when an error occurs, recursively printing submodules.

        Parameters
        ----------
        error : Exception
            The exception that was caught.
        context : str, default ""
            Additional context string to print.
        recursive : bool, default True
            If True, recursively print debug info for all submodules.
        """
        print("\n" + "!" * 80, file=sys.stderr)
        print(f"  ERROR OCCURRED: {type(error).__name__}", file=sys.stderr)
        print("!" * 80, file=sys.stderr)
        
        if context:
            print(f"\nContext: {context}\n", file=sys.stderr)
        
        print(f"Error message: {str(error)}\n", file=sys.stderr)
        
        # Print traceback
        print("Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        
        # Print debug summary for this module
        self.print_debug_summary(
            title=f"{self.__class__.__name__} State at Error",
            file=sys.stderr
        )
        
        # Recursively print debug summaries for submodules
        if recursive:
            self._print_recursive_debug_summaries(file=sys.stderr)
    
    def _print_recursive_debug_summaries(self, file=sys.stderr, visited=None, indent_level=0):
        """
        Recursively print debug summaries for all submodules.

        Parameters
        ----------
        file : file-like, default sys.stderr
            File to write output to.
        visited : set, optional
            Set of already visited module ids (to avoid infinite recursion).
        indent_level : int, default 0
            Current indentation level for nested modules.
        """
        if visited is None:
            visited = set()
        
        # Avoid infinite recursion
        if id(self) in visited:
            return
        visited.add(id(self))
        
        # Find all relevant submodules and attributes to debug
        debug_attrs = []
        
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
            
            try:
                attr_value = getattr(self, attr_name)
                
                # Check if it's a debuggable module or object
                if self._is_debuggable(attr_value):
                    debug_attrs.append((attr_name, attr_value))
            except Exception:
                continue
        
        # Print debug info for each relevant attribute
        for attr_name, attr_value in sorted(debug_attrs):
            indent = "  " * indent_level
            
            # Print section header
            print(f"\n{indent}{'▼' * 40}", file=file)
            print(f"{indent}  SUBMODULE: {attr_name}", file=file)
            print(f"{indent}{'▼' * 40}", file=file)
            
            # Print debug summary
            if hasattr(attr_value, 'print_debug_summary'):
                attr_value.print_debug_summary(
                    title=f"{attr_name} ({attr_value.__class__.__name__})",
                    file=file
                )
                
                # Recurse into submodule if it has the recursive method
                if hasattr(attr_value, '_print_recursive_debug_summaries'):
                    attr_value._print_recursive_debug_summaries(
                        file=file, 
                        visited=visited, 
                        indent_level=indent_level + 1
                    )
            else:
                # Fallback for objects without debug mixin
                print_module_summary(attr_value, title=f"{attr_name}", file=file)
    
    def _is_debuggable(self, obj):
        """
        Check if an object should be included in recursive debugging.

        Parameters
        ----------
        obj : any
            Object to check.

        Returns
        -------
        bool
            True if object should be debugged recursively.
        """
        # Include torch modules
        if isinstance(obj, torch.nn.Module):
            return True
        
        # Include objects with custom debug capabilities
        if hasattr(obj, 'print_debug_summary'):
            return True
        
        # Include specific types we want to debug
        debuggable_types = (
            'ReflectionData',
            'Model', 
            'ModelFT',
            'Scaler',
            'Restraints',
            'SolventModel',
        )
        
        return obj.__class__.__name__ in debuggable_types

def print_module_summary(module, title: str = None, file=sys.stderr):
    """
    Print debug summary for any module.

    Standalone function to print debug information for modules that
    may or may not have the DebugMixin.

    Parameters
    ----------
    module : object
        The module to inspect.
    title : str, optional
        Title for the summary.
    file : file-like, default sys.stderr
        File to write output to.
    """
    if hasattr(module, 'print_debug_summary'):
        module.print_debug_summary(title=title, file=file)
    else:
        # Fallback for modules without the mixin
        if title is None:
            title = f"{module.__class__.__name__} Summary"
        
        print("\n" + "=" * 80, file=file)
        print(f"  {title}", file=file)
        print("=" * 80, file=file)
        print(f"  Type: {type(module)}", file=file)
        print(f"  Attributes: {len(dir(module))}", file=file)
        
        # Try to get basic info
        if isinstance(module, torch.nn.Module):
            try:
                n_params = sum(p.numel() for p in module.parameters())
                print(f"  Parameters: {n_params:,}", file=file)
            except:
                pass
            
            try:
                device = next(module.parameters()).device
                print(f"  Device: {device}", file=file)
            except:
                pass
        
        print("=" * 80, file=file)
