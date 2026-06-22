# Crafter Code Policy DEO Hillclimb

This ReportBench lane asks an agent to improve a pure symbolic Crafter policy.
The worker receives a deterministic baseline policy, a local sweep runner, and
the same HillClimbSymbolicBench workproduct contract used by Craftax, NetHack,
and DungeonGrid.

The policy must be ordinary Python code. It may inspect symbolic game state
from Crafter, but it must not call an LLM, load a trained model, or use network
I/O while choosing actions.
