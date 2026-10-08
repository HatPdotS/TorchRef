{#- autosummary resolves a dotted name by attribute lookup first, so a submodule shadowed
    by a re-exported function of the same name reaches this template as that function.
    Those listed here get automodule, which imports by module path; every other name
    gets autosummary's stock function page. -#}
{%- set shadowed_modules = [
    "torchref.base.fourier.fft",
    "torchref.base.metrics.binwise_scale",
    "torchref.base.metrics.rfactor",
] -%}
{{ fullname | escape | underline }}
{% if fullname in shadowed_modules %}
.. automodule:: {{ fullname }}
{% else %}
.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}
{% endif %}
