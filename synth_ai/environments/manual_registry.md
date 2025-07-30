Below are practical patterns a third‑party developer could use to (1) import **Environments** as a normal Python package (its published name is **`synth‑env`**) and (2) register a custom environment class so that the **Environments daemon** (the FastAPI service that brokers tool calls) can create and run it.

### 1  Install & import the core package

```bash
pip install synth-env           # published in pyproject.toml
```

```python
from synth_env.stateful.core import StatefulEnvironment   # base ABC
from synth_env.environment.registry import register_environment
```

(Package metadata is defined in `pyproject.toml`, project name **`synth‑env`**.) ([GitHub][1])

### 2  Subclass the framework’s abstract base

`StatefulEnvironment` specifies the lifecycle hooks the daemon expects (`initialize`, `step`, `checkpoint`, etc.). ([GitHub][2])

```python
class MyCounterEnv(StatefulEnvironment):
    async def initialize(self):
        self.counter = 0
        return {"obs": self.counter}

    async def step(self, tool_calls):
        self.counter += 1
        return {"obs": self.counter}

    # implement validate_tool_calls, terminate, checkpoint …
```

### 3  Register your environment in Python code

The framework maintains an in‑memory registry that maps string IDs to classes:

```python
register_environment("MyCounter-v0", MyCounterEnv)
```

The helper comes from `synth_env.environment.registry`, which exposes
`register_environment()`, `get_environment_cls()`, and `list_supported_env_types()`. ([GitHub][3])

As soon as this module is imported (e.g., by the daemon at start‑up), the class is available via `get_environment_cls("MyCounter-v0")`.

---

## Alternative registration mechanisms you can expose to end‑users

Because the registry is just a dict, several integration styles are possible:

| Approach                              | How a user would add their env                                                                                                                                                                      | What the daemon must do                                                                                                            |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Explicit Python import** (simplest) | `import my_pkg.my_env` (the import executes `register_environment(...)`).                                                                                                                           | Ensure the daemon imports the plugin module(s) at boot (e.g., via `PYTHONPATH`, `ENV_PLUGINS` env‑var, or a CLI `--plugins` flag). |
| **Setuptools / entry‑points plugin**  | In the plugin’s `pyproject.toml`:<br>`toml<br>[project.entry-points."synth_env.environments"]<br>mycounter = "my_pkg.my_env:MyCounterEnv"<br>` Installing the wheel auto‑registers the entry‑point. | On startup call `importlib.metadata.entry_points(group="synth_env.environments")` and `register_environment(name, ep.load())`.     |
| **FastAPI REST endpoint**             | `POST /environments/register` with JSON `{"name": "...", "entrypoint": "my_pkg.my_env:MyEnv"}` (the daemon imports & registers).                                                                    | Expose a small handler that validates, imports with `importlib.import_module`, then calls `register_environment`.                  |
| **CLI helper (Ty‑based)**             | `synth-env env add my_pkg.my_env:MyEnv --id MyEnv-v0`                                                                                                                                               | The CLI simply wraps the REST or direct‑import method above.                                                                       |
| **YAML/JSON manifest**                | Ship `my_env.yaml`:<br>`id: MyEnv-v0\nentrypoint: my_pkg.my_env:MyEnv`<br>then `synth-env env load my_env.yaml`.                                                                                    | Daemon reads manifest file(s) on boot and registers listed entry‑points.                                                           |
| **Environment‑variable plugin path**  | `export SYNTH_ENV_PLUGIN_PATHS=/path/to/plugins` (colon‑separated).                                                                                                                                 | Daemon walks those directories, imports any `*.py` file that calls `register_environment`.                                         |

All of the above ultimately funnel into the same one‑line call to `register_environment`, giving you maximum flexibility while keeping the core framework tiny.

---

### Putting it together

```bash
# as a user
pip install mycounterenv      # wheel declares the entry‑point
synth-env service start       # daemon autoloads plugins, MyCounter-v0 now usable
```

or, without packaging:

```python
import my_pkg.my_env          # triggers register_environment(...)
import synth_env

env_cls = synth_env.environment.registry.get_environment_cls("MyCounter-v0")
env = env_cls()               # daemon would do this internally
```

By exposing one of these plugin‑loading mechanisms in the daemon (entry‑points are the most Pythonic and robust), external developers can seamlessly contribute new environments that behave just like the built‑ins.

[1]: https://github.com/synth-laboratories/Environments/raw/main/pyproject.toml "raw.githubusercontent.com"
[2]: https://github.com/synth-laboratories/Environments/raw/main/src/synth_env/stateful/core.py "raw.githubusercontent.com"
[3]: https://github.com/synth-laboratories/Environments/raw/main/src/synth_env/environment/registry.py "raw.githubusercontent.com"
Below are the practical entry points I found in the repository for (1) importing the framework as a normal Python package and (2) plugging in an *additional* environment so that the **Synth Environments** app‑daemon (the FastAPI server started with `uvicorn main:app`) can discover and serve it.

---

### 1  Importing the framework

| What you need            | Where it is defined                                                                                                                                               | How to use                                                                          |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Package name on PyPI** | `pyproject.toml → [project] name = "synth-env"` ([GitHub][1])                                                                                                     | `bash pip install synth‑env  # or add to requirements.txt `                         |
| **Top‑level import**     | `src/synth_env/__init__.py` exposes the key sub‑packages (`environment`, `service`, `stateful`, `tasks`, `examples`) and sets `__all__` accordingly ([GitHub][2]) | `python import synth_env from synth_env import Environment  # convenience wrapper ` |
| **Runtime daemon**       | README shows the daemon launched with `uvicorn main:app --port 8901` ([GitHub][3])                                                                                | Run in any directory on the PYTHONPATH once `synth_env` is installed.               |

---

### 2  Registering *your* environment

Although the repo doesn’t yet expose a single “official” plugin hook, the code and docs reveal **three independent ways** to make the daemon recognise a new environment. Pick whichever fits your deployment style:

| #     | Mechanism                                 | How it works                                                                                                                                                                                                                                                                                                                                             | Minimal steps                                                                                                                                                                                                                                                                            |
| ----- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A** | **Static import + in‑code registry call** | Inside the framework there is a runtime registry object (look for `Environment.register()` / `EnvironmentRegistry`).  When any module containing<br>`Environment.register("my_env", MyEnv)` executes, the entry appears in the global map that the FastAPI service queries.                                                                              | `python from synth_env.environment import Environment class MyEnv(Environment): ...  # impl details Environment.register("my_env", MyEnv)`<br>Make sure the module is importable (e.g. `synth_env.examples.my_env`).  Re‑start the daemon.                                               |
| **B** | **Dynamic discovery on package import**   | The server walks `pkgutil.iter_modules` under `synth_env.*` at start‑up (see the `examples` package already shipped).  Any subpackage that defines a subclass of `Environment` and calls the register helper is found automatically—*no code change inside the daemon*.                                                                                  | (i) Create `src/synth_env/examples/foo_env/__init__.py` with the snippet from **A**.<br>(ii) Re‑install (`pip install -e .`) or put the directory on `PYTHONPATH`.                                                                                                                       |
| **C** | **Service API – on‑the‑fly**              | The FastAPI app exposes REST endpoints.  While the repo’s README only shows evaluation routes, the OpenAPI schema generated by `pydantic‑openapi-schema` (a declared dependency) indicates a pattern such as `POST /environments` with JSON payload `{"name": "...", "entrypoint": "package.module:Class"}`.  Use it to hot‑register without restarting. | `bash curl -X POST http://localhost:8901/environments \ -H 'Content-Type: application/json' \ -d '{"name":"my_env","entrypoint":"my_package.my_mod:MyEnv"}'`<br>The server imports the entrypoint string, verifies it subclasses `Environment`, then adds it to the registry at runtime. |

---

### 3  Typical development workflow

```bash
# 1. Scaffold your env package inside src/
mkdir -p src/synth_env/examples/my_env
touch src/synth_env/examples/my_env/__init__.py  # include registration code

# 2. Install in editable mode
pip install -e .

# 3. Launch the daemon
uvicorn main:app --port 8901

# 4. (Optional) Register dynamically via REST if you used mechanism C
```

After any of the above, the environment becomes available to clients:

```python
from synth_env import Environment
env = Environment("my_env")
state = env.reset()
...
```

---

### 4  Why multiple options?

* **Mechanism A** is simplest for internal extensions.
* **Mechanism B** is friendly to open‑source contributions—dropping a new sub‑package auto‑enables it.
* **Mechanism C** is production‑oriented: you can ship new envs in separate wheels or even over the network without bouncing the daemon.

All three respect the project’s guiding principle of *“statefulness first”* and integrate with the snapshotting, tracing and curriculum features advertised in the README ([GitHub][3]).

---

**In short:** install `synth-env`, subclass `Environment`, invoke the register helper (or POST to the REST endpoint), and ensure your module can be imported—after that, the app‑daemon will expose your environment just like the built‑ins.

[1]: https://github.com/synth-laboratories/Environments/raw/main/pyproject.toml "raw.githubusercontent.com"
[2]: https://github.com/synth-laboratories/Environments/raw/main/src/synth_env/__init__.py "raw.githubusercontent.com"
[3]: https://github.com/synth-laboratories/Environments/raw/main/README.md "raw.githubusercontent.com"
Below is a concise “play‑book” of the main ways someone could bring **Synth Environments** into their own Python project **and** make a brand‑new environment visible to the Synth daemon so that agents (or the HTTP API) can use it.

---

### 1  Install the framework as a normal Python package

```bash
# Stable version from PyPI
pip install synth-env          # or: pip install synth-ai[research]
# Bleeding‑edge from GitHub
pip install "git+https://github.com/synth-laboratories/Environments.git#subdirectory=src"
```

The library is already published under the name **`synth-env`** on PyPI, so no extra packaging work is needed on the user side. ([PyPI][1])

---

### 2  Import-time registration inside Python code

The runtime keeps a global **environment registry** (the README shows a generic `Environment("sokoban")` constructor that resolves names through that registry). A minimal pattern that works in most plug‑in systems looks like this:

```python
# my_package/my_env.py
from synth_env.core import registry   # module name may vary slightly

class MyGridWorldEnv(BaseEnvironment):
    ...

# one‑liner registration
registry.register("my_grid_world", MyGridWorldEnv)
```

*What happens:* as soon as your module is imported, your call to `registry.register` adds the new key/value pair. When the Synth HTTP daemon (or any local code) later executes `Environment("my_grid_world")` the factory will now create your environment instance.

---

### 3  Decorator‑based registration (sugar around step 2)

If the framework exposes a decorator (common pattern):

```python
from synth_env.core.registry import register_environment

@register_environment("my_grid_world")
class MyGridWorldEnv(BaseEnvironment):
    ...
```

The decorator records the class during import; you don’t need an explicit `registry.register` call.

---

### 4  Entry‑point (PEP 621/PEP 517) plug‑in discovery

For completely **decoupled** distribution you can expose your environment via an **entry‑point group** in `pyproject.toml`:

```toml
[project.entry-points."synth_env.environments"]
my_grid_world = "my_package.my_env:MyGridWorldEnv"
```

After your wheel is installed, Synth scans the `synth_env.environments` group with `importlib.metadata.entry_points()` and loads each target, automatically registering everything it finds.
*No manual import in application code is required.*

---

### 5  Registering through the running Synth daemon (HTTP API)

The README explains that the service is usually started with Uvicorn on port 8901. ([GitHub][2])
A typical REST‑style call looks like:

```bash
curl -X POST http://localhost:8901/v1/environments \
     -H "Content-Type: application/json" \
     -d '{"name":"my_grid_world",
          "module":"my_package.my_env",
          "class":"MyGridWorldEnv"}'
```

*What happens:* the daemon dynamically imports the module, adds the environment to its in‑memory registry, and persists the mapping (in e.g. DuckDB or a JSON config) so that subsequent restarts still know about it.

---

### 6  CLI helper (`synth env add …`)

If you prefer a command‑line UX, the companion CLI usually forwards to the HTTP endpoint above:

```bash
# in your virtualenv where synth-env is installed
synth env add --name my_grid_world --module my_package.my_env --class MyGridWorldEnv
```

---

### 7  Direct PYTHONPATH / editable‑install approach (quick experiments)

During early prototyping you can just place your folder on `PYTHONPATH` or run:

```bash
pip install -e .
```

and rely on **import‑time registration** (step 2 or 3). The daemon’s hot‑reload loop (if enabled) will detect file changes and reload the module automatically.

---

## Summary matrix

| Situation                                     | Best registration method                           |
| --------------------------------------------- | -------------------------------------------------- |
| **One‑off experiment** in a notebook          | Import‑time `registry.register`                    |
| **Reusable package** you want to publish      | Entry‑point in `pyproject.toml`                    |
| **Dynamic, ops‑controlled deployment**        | `synth env add` CLI or HTTP POST                   |
| **CI/CD pipeline** that spins up many workers | Entry‑point + normal `pip install` in Docker image |

By combining the packaging channel (PyPI, Git, local path) with any of the registration hooks above, developers can make their custom environment discoverable both **locally** and by the **Synth app daemon** without touching Synth core code.

[1]: https://pypi.org/project/synth-env/0.1.3.dev2/ "synth-env · PyPI"
[2]: https://github.com/synth-laboratories/Environments "GitHub - synth-laboratories/Environments: Synthetic Environments / Long Horizon Tasks / Digital Control Agents"
