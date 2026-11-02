"""Instantiate explicitly configured, trusted integration factories."""
import importlib


def load_factory(specification):
    module, separator, factory = specification.partition(":")
    if not separator or not module or not factory:
        raise ValueError("Integration factory must use MODULE:FACTORY")
    return getattr(importlib.import_module(module), factory)()
