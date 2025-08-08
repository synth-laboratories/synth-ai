import sys as _sys
import importlib as _importlib

# Forward top-level package
_pkg = _importlib.import_module('synth_ai.v0.tracing_v1')
_sys.modules[__name__] = _pkg

# Explicitly forward submodules so `synth_ai.tracing_v1.X` works
_SUBMODULES = [
    'abstractions',
    'base_client',
    'client_manager',
    'config',
    'context',
    'decorators',
    'immediate_client',
    'local',
    'log_client_base',
    'retry_queue',
    'trackers',
    'upload',
    'utils',
]
for _m in _SUBMODULES:
    _sys.modules[f'{__name__}.{_m}'] = _importlib.import_module(f'synth_ai.v0.tracing_v1.{_m}')

# Forward events package and its submodules
_events_pkg = _importlib.import_module('synth_ai.v0.tracing_v1.events')
_sys.modules[f'{__name__}.events'] = _events_pkg
for _m in ['manage', 'scope', 'store']:
    _sys.modules[f'{__name__}.events.{_m}'] = _importlib.import_module(f'synth_ai.v0.tracing_v1.events.{_m}')
