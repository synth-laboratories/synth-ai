"""Compatibility aliases retained for one release."""

import functools
import warnings


def artifact_alias(method):
    @functools.wraps(method)
    def call(*args, **kwargs):
        warnings.warn(
            "swarms.files and swarms.outputs are deprecated; use artifacts and run.produced().",
            DeprecationWarning,
            stacklevel=2,
        )
        return method(*args, **kwargs)

    return call
