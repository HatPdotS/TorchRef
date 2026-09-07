# Fixture ownership

The root `tests/conftest.py` owns pytest options, markers, capability gating,
and the `pytest_plugins` registry. Put reusable setup in the modules below.
Keep a fixture in its test module when only that module needs it.

| Module | Responsibility | Visibility / lifetime |
|---|---|---|
| `paths.py` | Repository, bundled-data and optional library paths | All tests; session |
| `files.py` | Sample-file selection and matching structure pairs | All tests; session; no model loading |
| `devices.py` | Configured device, explicit backends, device parametrization | All tests; existing per-fixture scopes |
| `precision.py` | Comparison tolerances and CPU-double reference context | All tests; reference fixture restores state after each test |
| `objects.py` | Mutable models, data, scalers and restraints | All tests; fresh per test except explicitly shared bundles |
| `numerical.py` | Synthetic tensors and factories | Imported only by `unit/conftest.py`; function |
| `functional.py` | `shared_model`, `shared_model_ft`, `shared_reflection_data` | Imported only by `functional/conftest.py`; module |

Existing fixture names remain available without imports in tests. Import reusable
helpers from their defining module, never from the root `conftest.py`. Subtree
conftests import fixture functions explicitly; register shared plugins only at
the root so pytest also works when invoked from a subdirectory.

Use `shared_*` only for read-only checks. They capture the package configuration
at module setup and may populate derived caches. Do not move them, change their
parameters, tables, masks or grids, backpropagate through them, or use them in
tests that switch global configuration. A target or scaler can mutate a model it
borrows, so a shared model must not be passed to such an operation.

Tests that verify loading must invoke the loader themselves. Tests of mutation,
device movement, or empty caches use fresh objects. `loaded_model`,
`loaded_model_ft`, `loaded_reflection_data`, and their composed fixtures in `objects.py` provide
fresh mutable objects per test. The explicitly shared session bundles in that
module retain their documented ownership contracts.

Use `cpu_double_precision()` to scope an explicit numerical reference, or request
`double_cpu` for a single test. The structure-factor package uses the same context
at package scope; both usages restore dtype, device, and density cutoff on exit.

This separation preserves the existing numerical-factory allocation policy and
test-selection policy. Those policies are independent of fixture registration and
scope, and can be revised in their respective modules.
