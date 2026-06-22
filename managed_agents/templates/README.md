# Managed Agents Verifier Templates

These templates are the launch-era authoring surface for verifier configs that
**Managed Research** can consume easily from `synth-ai`.

The intended hierarchy is:

- **Managed Research** is the service/product.
- **Managed Agents** is supporting **beta infrastructure**.
- Verifier configs describe how managed-agents-backed verifier/backend flows
  should behave for a concrete research task.

## What lives here

- `verifier_config.template.yaml` is the canonical general template.
- Task-specific verifier examples should layer on top of that template rather
  than invent a new shape.

## What the template covers

The single YAML template is meant to keep the launch-era authoring surface small
and understandable. It includes:

- verifier identity and purpose
- launch/product positioning
- visible evidence fields
- hidden eval-only fields
- prompt and instruction sections
- result/output contract
- environment/action constraints
- notes on how Managed Research consumes the config

## How to use it

1. Copy `verifier_config.template.yaml`.
2. Fill in the task objective, evidence sources, instructions, and result schema.
3. Keep Managed Agents framed as verifier/backend infrastructure, not the public
   product headline.
4. Add one task-specific example alongside the template when the verifier is
   important enough to explain publicly or operationally.

## Launch default

For this launch push, keep the structure intentionally minimal:

- one general template
- one README
- one concrete example

Do not split the authoring surface into multiple schemas or frameworks unless
launch work proves the single-template shape is insufficient.
