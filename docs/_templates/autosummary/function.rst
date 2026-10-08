{#- Only the recursive "Modules" lists generate function pages. autosummary resolves a
    dotted name by attribute lookup first, so a submodule shadowed by a re-exported
    function of the same name (torchref.base.fourier.fft) arrives here as that
    function. automodule imports by module path and documents the submodule itself. -#}
{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
