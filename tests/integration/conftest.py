"""Use shared fixtures registered by the root conftest for integration tests.

Pipeline-specific fixtures belong in their consuming modules. Mutable loaded
objects from ``tests.fixtures.objects`` are function-scoped unless documented
as explicitly shared.
"""
