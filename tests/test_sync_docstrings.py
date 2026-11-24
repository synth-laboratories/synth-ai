"""Synchronous reference prose must follow its authored async implementation."""

import inspect

from synth_ai import _sync, account, billing, swarms


def test_generated_delegates_preserve_public_method_documentation() -> None:
    for module in (account, billing, swarms):
        for name, source in vars(module).items():
            if not inspect.isclass(source) or not name.startswith("Async"):
                continue
            if source.__module__ != module.__name__:
                continue
            target = getattr(_sync, name.removeprefix("Async") + "Client")
            for method_name, method in vars(source).items():
                if method_name.startswith("_") or not (
                    inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method)
                ):
                    continue
                documentation = inspect.getdoc(method)
                assert documentation is not None and documentation != "", (
                    f"{name}.{method_name} lacks public documentation"
                )
                assert inspect.getdoc(getattr(target, method_name)) == documentation
