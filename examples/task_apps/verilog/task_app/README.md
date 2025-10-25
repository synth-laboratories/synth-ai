# GRPO Verilog Task App

This example mirrors the Crafter task app layout while targeting the Verilog
hardware synthesis environment under `synth_ai.environments.examples.verilog`.
The `grpo_verilog.py` module builds a lightweight dataset from the VerilogEval
spec-to-RTL benchmark and wires a minimalist task-app configuration. The
companion `grpo_verilog_task_app.py` acts as a compatibility wrapper for direct
FastAPI execution or Modal deployment.

The rollout bridge currently surfaces the initial observation for the selected
task instance, providing a scaffold for future extensions that integrate the
full hosted environment workflow and policy orchestration similar to Crafter.
