### Quickstart (using config.toml)

The defaults come from `examples/finetuning/synth_qwen/config.toml`. Override with env vars only when needed.

#### 1) Rollouts → v3 traces
```bash
set -a; source .env 2>/dev/null || true; set +a
uvpm examples.finetuning.synth_qwen.run_crafter_qwen4b
```

Notes:
- Model, episodes, steps, difficulty, temperature, tool choice, etc. are taken from `config.toml`.
- Only API key and v3 db path are specified here.

Example output (abridged):
```text
✅ Crafter service is healthy
🚀 Running 10 episodes (concurrency=5)...
✅ Completed 10 episodes in ~366s
📊 EVALUATION RESULTS
Episodes completed: 10/10
Average reward per episode: 1.10
Average steps per episode: 87.00
💾 Results: traces/v3/synth_ai.db
```

#### 2) Filter traces → SFT JSONL
```bash
uvpm examples.finetuning.synth_qwen.filter_traces_achievements
```

Notes:
- Filter settings (achievements, thresholds, output path) are defined in `config.toml`.

Example output:
```text
Using database: sqlite+aiosqlite:///$PWD/traces/v3/synth_ai.db/dbs/default/data
Output file: ft_data/qwen4b_crafter_sft_collect_wood.jsonl
✅ Wrote 13 examples from 13 sessions
```

#### 3) Kick off SFT (prod)
```bash
set -a; source .env 2>/dev/null || true; set +a
uvpm examples.finetuning.synth_qwen.sft_kickoff
```

Notes:
- Base model and training JSONL path come from `config.toml`.

Example output (abridged):
```text
🚀 Starting Qwen 4B SFT
⏳ poll ...
🟢 Qwen4B SFT fine-tune succeeded → ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-22
```

#### 4) Rollouts with fine-tuned model
```bash
set -a; source .env 2>/dev/null || true; set +a
CRAFTER_MODEL="ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-22" uvpm examples.finetuning.synth_qwen.run_crafter_qwen4b
```

Notes:
- Replace `ftjob-22` with the job id printed by your SFT step.
- If you see 401 Invalid API key, export the prod key: `export SYNTH_API_KEY="$SYNTH_API_KEY_PROD"`.

Example output (abridged):
```text
✅ Model warmed up successfully!
🚀 Running 10 episodes (concurrency=5)...
✅ Completed 10 episodes in ~480s
📊 EVALUATION RESULTS
Episodes completed: 10/10
Average reward per episode: 1.60
Average steps per episode: 90.80
Achievements: collect_wood in 6/10 episodes
💾 Results: traces/v3/synth_ai.db
```



```
joshuapurtell@Mac synth-ai % bash examples/finetuning/synth_qwen/run_demo.sh
Synth Qwen4B finetuning demo (Crafter)

Run rollouts to generate v3 traces now? [Y/n]: Y
Using config defaults from examples/finetuning/synth_qwen/config.toml (override below if desired).
Model id [Enter=use config]: 
Episodes [Enter=use config]: 5
Max steps [Enter=use config]: 5
Difficulty [Enter=use config]: 
Enable think mode? (1/0) [Enter=0]: 

Running rollouts (v3 tracing)...
Detected SYNTH_API_KEY (sk_liv...ac95). Use this key? [Y/n]: N
Use SYNTH_API_KEY_PROD (sk_liv...a2a4)? [y/N]: y
[PATCH] Attempting to apply Crafter deterministic patch...
[PATCH] Patching crafter.Env._balance_object...
[PATCH] crafter.Env._balance_object patched.
[PATCH] Attempting to apply Crafter serialization patch v3...
[PATCH] Adding enhanced save/load methods to crafter.Env...
[PATCH] crafter.Env.save() and load() methods added (v3).
[PATCH] Crafter serialization patch v3 complete.
[PATCH] Attempting to apply simplified Crafter world configuration patch...
[PATCH] Simplified Crafter world configuration patch complete.
[PATCH] Available configs: easy, normal, hard, peaceful
🔧 Using Synth base URL = https://agent-learning.onrender.com/api
🔇 Quiet mode enabled - suppressing verbose logs
✅ Crafter service is healthy: {'status': 'ok', 'supported_environments': ['CrafterClassic', 'CrafterCustom']}

🔥 Warming up Qwen/Qwen3-4B-Instruct-2507 on Synth backend...
✅ Warmed Qwen/Qwen3-4B-Instruct-2507 in 3s        arming elapsed=0s
✅ Model warmed up successfully!

🚀 Starting sqld daemon for v3 tracing...
✅ sqld daemon started

📊 V3 Tracing enabled. Traces will be saved to: traces/v3/synth_ai.db
   Experiment: crafter_lm_synth_Qwen/Qwen3-4B-Instruct-2507_20250808_165304

🚀 Running 5 episodes (concurrency=5)...

📤 Starting episodes...
Episode 1: 100%|███████████████| 5/5 [00:18<00:00,  3.71s/it, tc=1, act=3, tok=78, in=860, tps=22.4]
Episode 0: 100%|██████████████| 5/5 [00:19<00:00,  3.89s/it, tc=1, act=3, tok=94, in=861, tps=22.79]
Episode 2: 100%|██████████████| 5/5 [00:18<00:00,  3.74s/it, tc=1, act=2, tok=73, in=832, tps=21.06]
Episode 4: 100%|██████████████| 5/5 [00:19<00:00,  3.90s/it, tc=1, act=3, tok=91, in=818, tps=17.34]
Episode 3: 100%|██████████████| 5/5 [00:20<00:00,  4.15s/it, tc=1, act=2, tok=86, in=859, tps=21.96]

✅ Completed 5 episodes in 23.57 seconds

==================================================
📊 EVALUATION RESULTS
==================================================
Episodes completed: 5/5
Failed episodes: 0
Total reward: 1.00
Average reward per episode: 0.20
Total steps: 68
Average steps per episode: 13.60

Seeds used:
  Episode 0: seed 1
  Episode 1: seed 2
  Episode 2: seed 3
  Episode 3: seed 4
  Episode 4: seed 5
Unique achievements unlocked: 1

Achievements unlocked:
  - collect_sapling: 1 episodes (20.0%)

Action counts (total: 68):
  - move_right: 25 (36.8%)
  - do: 19 (27.9%)
  - move_down: 10 (14.7%)
  - move_left: 5 (7.4%)
  - place_stone: 3 (4.4%)
  - move_up: 3 (4.4%)
  - make_wood_sword: 1 (1.5%)
  - make_stone_pickaxe: 1 (1.5%)
  - make_wood_pickaxe: 1 (1.5%)

💾 Results available in Turso database: traces/v3/synth_ai.db
   Experiment ID: exp_a0b091fd12c6
   Use the filter_traces_sft_turso.py script to extract fine-tuning data

Markdown row:
| Qwen/Qwen3-4B-Instruct-2507 |       5 |     0.20 |         0.020 |        0.001 |       68.000 |   13.600 |

✅ Stopped sqld daemon

Filter v3 traces into SFT JSONL now? [Y/n]: Y
Using DB: sqlite+aiosqlite:////Users/joshuapurtell/Documents/GitHub/synth-ai/traces/v3/synth_ai.db/dbs/default/data
You can override filter options; Enter to use config defaults.
Required achievements (space-separated) [Enter=config]: 
Restrict to models (space-separated) [Enter=all]: 
Output JSONL path [Enter=config]: 
Min total reward [Enter=config]: 1
Max total cost [Enter=config]: 
Max total tokens [Enter=config]: 

Filtering traces to SFT JSONL...
[PATCH] Attempting to apply Crafter deterministic patch...
[PATCH] Patching crafter.Env._balance_object...
[PATCH] crafter.Env._balance_object patched.
[PATCH] Attempting to apply Crafter serialization patch v3...
[PATCH] Adding enhanced save/load methods to crafter.Env...
[PATCH] crafter.Env.save() and load() methods added (v3).
[PATCH] Crafter serialization patch v3 complete.
[PATCH] Attempting to apply simplified Crafter world configuration patch...
[PATCH] Simplified Crafter world configuration patch complete.
[PATCH] Available configs: easy, normal, hard, peaceful
🤖 Modal/Synth FT Filter (achievements)
Using database: sqlite+aiosqlite:////Users/joshuapurtell/Documents/GitHub/synth-ai/traces/v3/synth_ai.db/dbs/default/data
Output file: ft_data/qwen4b_crafter_sft.jsonl
Filters: {
  "required_achievements": [
    "collect_wood"
  ],
  "models": [
    "Qwen/Qwen3-4B-Instruct-2507"
  ],
  "min_total_reward": 1.0,
  "max_cost": 10.0,
  "max_tokens": 100000
}

✅ Wrote 23 examples from 23 sessions

Kick off SFT training job now? [Y/n]: Y
Enter overrides for training job; Enter to use config.
Base model [Enter=config]: 
Training JSONL path [Enter=config]: 

Starting SFT job...
Detected SYNTH_API_KEY (sk_liv...a2a4). Use this key? [Y/n]: n
Use SYNTH_API_KEY_PROD (sk_liv...a2a4)? [y/N]: Y
🚀 Starting Qwen 4B SFT
⏳ poll 1/20 – status = queued
⏳ poll 2/20 – status = queued
⏳ poll 3/20 – status = queued
⏳ poll 4/20 – status = queued
⏳ poll 5/20 – status = queued
⏳ poll 6/20 – status = queued
⏳ poll 7/20 – status = queued
⏳ poll 8/20 – status = queued
⏳ poll 9/20 – status = running
⏳ poll 10/20 – status = queued
⏳ poll 11/20 – status = queued
⏳ poll 12/20 – status = succeeded
🟢 Qwen4B SFT fine-tune succeeded → ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-6cedf721e0ca4c80968834b71e2bdace
⏱️ wall-clock: 249.3s | trained_tokens: 41777
Captured fine-tuned model id: ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-6cedf721e0ca4c80968834b71e2bdace
SFT logs saved to: logs/sft_kickoff_20250808_165436.log

Roll out fine-tuned model 'ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-6cedf721e0ca4c80968834b71e2bdace' in Crafter now? [y/N]: Y
Episodes [Enter=config]: 5
Max steps [Enter=config]: 5
Difficulty [Enter=config]: 
Enable think mode? (1/0) [Enter=0]: 

Running rollouts with fine-tuned model...
[PATCH] Attempting to apply Crafter deterministic patch...
[PATCH] Patching crafter.Env._balance_object...
[PATCH] crafter.Env._balance_object patched.
[PATCH] Attempting to apply Crafter serialization patch v3...
[PATCH] Adding enhanced save/load methods to crafter.Env...
[PATCH] crafter.Env.save() and load() methods added (v3).
[PATCH] Crafter serialization patch v3 complete.
[PATCH] Attempting to apply simplified Crafter world configuration patch...
[PATCH] Simplified Crafter world configuration patch complete.
[PATCH] Available configs: easy, normal, hard, peaceful
🔧 Using Synth base URL = https://agent-learning.onrender.com/api
🔇 Quiet mode enabled - suppressing verbose logs
✅ Crafter service is healthy: {'status': 'ok', 'supported_environments': ['CrafterClassic', 'CrafterCustom']}

🔥 Warming up ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-6cedf721e0ca4c80968834b71e2bdace on Synth backend...
⏳ Warming ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-6cedf721e0ca4c80968834b71e2bdace [|] status=timeout elapsed=10⏳ Warming ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-6cedf721e0ca4c80968834b71e2bdace [/] status=timeout elapsed=21✅ Warmed ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-6cedf721e0ca4c80968834b71e2bdace in 22s        
✅ Model warmed up successfully!

🚀 Starting sqld daemon for v3 tracing...
✅ sqld daemon started

📊 V3 Tracing enabled. Traces will be saved to: traces/v3/synth_ai.db
   Experiment: crafter_lm_synth_ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-6cedf721e0ca4c80968834b71e2bdace_20250808_165943

🚀 Running 5 episodes (concurrency=5)...

📤 Starting episodes...
Episode 2: 100%|██████████████| 5/5 [00:48<00:00,  9.80s/it, tc=1, act=3, tok=75, in=833, tps=15.24]
Episode 1: 100%|██████████████| 5/5 [00:52<00:00, 10.48s/it, tc=1, act=3, tok=73, in=840, tps=10.74]
Episode 3: 100%|██████████████| 5/5 [00:54<00:00, 10.88s/it, tc=1, act=3, tok=79, in=834, tps=15.59]
Episode 4: 100%|██████████████| 5/5 [00:54<00:00, 10.90s/it, tc=1, act=3, tok=75, in=817, tps=11.93]
Episode 0: 100%|███████████████| 5/5 [00:56<00:00, 11.38s/it, tc=1, act=2, tok=91, in=850, tps=5.53]

✅ Completed 5 episodes in 58.29 seconds

==================================================
📊 EVALUATION RESULTS
==================================================
Episodes completed: 5/5
Failed episodes: 0
Total reward: 3.00
Average reward per episode: 0.60
Total steps: 72
Average steps per episode: 14.40

Seeds used:
  Episode 0: seed 1
  Episode 1: seed 2
  Episode 2: seed 3
  Episode 3: seed 4
  Episode 4: seed 5
Unique achievements unlocked: 2

Achievements unlocked:
  - collect_sapling: 2 episodes (40.0%)
  - collect_wood: 1 episodes (20.0%)

Action counts (total: 72):
  - move_right: 25 (34.7%)
  - do: 19 (26.4%)
  - move_down: 8 (11.1%)
  - move_left: 8 (11.1%)
  - move_up: 6 (8.3%)
  - place_stone: 2 (2.8%)
  - place_table: 2 (2.8%)
  - make_wood_pickaxe: 1 (1.4%)
  - place_plant: 1 (1.4%)

💾 Results available in Turso database: traces/v3/synth_ai.db
   Experiment ID: exp_56d8e29cbf5a
   Use the filter_traces_sft_turso.py script to extract fine-tuning data

Markdown row:
| ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-6cedf721e0ca4c80968834b71e2bdace |       5 |     0.60 |         0.240 |        0.012 |       72.000 |   14.400 |

✅ Stopped sqld daemon

Done. You can re-run this script to repeat steps as needed.
joshuapurtell@Mac synth-ai % 
```