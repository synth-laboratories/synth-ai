joshuapurtell@Mac synth-ai % bash examples/evals/run_demo.sh                
Models to compare (space-separated) [gpt-5-nano gpt-4.1-nano]: 
Models: gpt-5-nano gpt-4.1-nano
Episodes per model [3]: 5
Max turns per episode [5]: 5
Parallelism per model (concurrency) [5]: 5
Difficulty [easy]: 
Running comparison: episodes=5, max_turns=5, difficulty=easy, concurrency=5
Detected SYNTH_API_KEY (sk_liv...ac95). Use this key? [Y/n]: n
Use SYNTH_API_KEY_PROD (sk_liv...a2a4)? [y/N]: Y
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
✅ Loaded 8 Crafter achievement hooks (Easy, Medium, Hard)
🎮 Crafter Multi-Model Experiment
==================================================
Experiment ID: crafter_multi_model_20250808_170152
Models: gpt-5-nano, gpt-4.1-nano
Episodes per model: 5
Max turns per episode: 5
Difficulty: easy
Seeds: 1000 to 1004
Turn timeout: 20.0s
Episode timeout: 180.0s
Save traces: True
Database URL: sqlite+aiosqlite:////Users/joshuapurtell/Documents/GitHub/synth-ai/traces/v3/synth_ai.db/dbs/default/data
==================================================
✅ Crafter service is running

Running 5 episodes for gpt-5-nano in parallel...

gpt-5-nano | ep1:   0%|                                                               | 0/5 [00:00<?, ?turn/s]
Running 5 episodes for gpt-4.1-nano in parallel...                                    | 0/5 [00:00<?, ?turn/s]
gpt-5-nano | ep3:   0%|                                                               | 0/5 [00:00<?, ?turn/s]
gpt-4.1-nano | ep3: 100%|██████████████████████████████████████████████| 5/5 [00:09<00:00,  1.95s/turn, ach=1]
gpt-4.1-nano | ep2:  80%|████████████████████████████████████▊         | 4/5 [00:10<00:02,  2.64s/turn, ach=2]
gpt-4.1-nano | ep4: 100%|██████████████████████████████████████████████| 5/5 [00:11<00:00,  2.32s/turn, ach=0]
gpt-4.1-nano | ep5: 100%|██████████████████████████████████████████████| 5/5 [00:11<00:00,  2.37s/turn, ach=2]
gpt-5-nano | ep1:  20%|█████████▌                                      | 1/5 [00:21<01:24, 21.13s/turn, ach=0    ⏰ Turn 3 timed out for episode 0 after 20.0s                       | 2/5 [00:25<00:38, 12.83s/turn, ach=0]
gpt-4.1-nano | ep1:  60%|███████████████████████████▌                  | 3/5 [00:28<00:19,  9.62s/turn, ach=1]
gpt-5-nano | ep3: 100%|████████████████████████████████████████████████| 5/5 [01:00<00:00, 12.05s/turn, ach=1]
gpt-5-nano | ep2: 100%|████████████████████████████████████████████████| 5/5 [01:07<00:00, 13.56s/turn, ach=2]
    ⏰ Turn 4 timed out for episode 3 after 20.0s██████████████████████| 5/5 [01:07<00:00, 14.04s/turn, ach=2]
gpt-5-nano | ep4:  80%|██████████████████████████████████████▍         | 4/5 [01:08<00:17, 17.02s/turn, ach=0]
gpt-5-nano | ep5: 100%|████████████████████████████████████████████████| 5/5 [01:13<00:00, 14.71s/turn, ach=1]
gpt-5-nano | ep1: 100%|████████████████████████████████████████████████| 5/5 [01:19<00:00, 15.83s/turn, ach=1]
gpt-4.1-nano | ep5: 100%|██████████████████████████████████████████████| 5/5 [00:11<00:00,  1.68s/turn, ach=2]
📊 Analysis Results:
================================================================================:13<00:00, 14.26s/turn, ach=1]

📈 Model Performance Summary:
Model                Avg Achievements   Max Achievements   Invalid Rate    Success Rate   
--------------------------------------------------------------------------------------
gpt-4.1-nano           1.20 ± 0.75                    2            0.00%          100.00%
gpt-5-nano             1.00 ± 0.63                    2            0.00%          100.00%

🏆 Achievement Frequencies:

Achievement                 gpt-4.1-nano   gpt-5-nano
-----------------------------------------------
collect_drink               2/5   ( 40%)   0/5   (  0%)
collect_sapling             1/5   ( 20%)   2/5   ( 40%)
collect_wood                3/5   ( 60%)   2/5   ( 40%)
place_plant                 0/5   (  0%)   1/5   ( 20%)

💰 Model Usage Statistics from Current Experiment:
Model                Provider   Usage Count  Avg Latency (ms)   Total Cost  
------------------------------------------------------------------------
gpt-5-nano           openai     221          13006.57           $0.0000     
gpt-4.1-nano         openai     161          950.12             $0.0000     

💾 Detailed results saved to: /Users/joshuapurtell/Documents/GitHub/synth-ai/temp/crafter_experiment_results_20250808_170312.json

✅ Experiment complete!
Using v3 traces DB: /Users/joshuapurtell/Documents/GitHub/synth-ai/traces/v3/synth_ai.db/dbs/default/data
\nAvailable achievements (session counts):
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
Achievements present (session counts):
  - collect_drink: 44
  - collect_sapling: 62
  - collect_wood: 74
  - defeat_skeleton: 4
  - defeat_zombie: 2
  - eat_cow: 2
  - place_plant: 8
  - place_table: 3
\nEnter achievements to filter by (space-separated), or press Enter for 'collect_wood':

Optionally restrict to models (space-separated), or press Enter to include all:

\nRunning: uv run python -m examples.evals.trace_analysis filter --db "/Users/joshuapurtell/Documents/GitHub/synth-ai/traces/v3/synth_ai.db/dbs/default/data" --achievements collect_wood --output ft_data/evals_filtered.jsonl
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
✅ Wrote 74 examples from 74 sessions → ft_data/evals_filtered.jsonl
\nRunning: uv run python -m examples.evals.trace_analysis stats --db "/Users/joshuapurtell/Documents/GitHub/synth-ai/traces/v3/synth_ai.db/dbs/default/data" --achievements collect_wood
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
Matched sessions (any of: collect_wood )
  n=74  avg_reward=0.76  stddev=1.00
  avg_first_unlock_step=4.7  stddev=4.6
Others
  n=224  avg_reward=0.21  stddev=0.51

Achievement frequency by session (matched vs others):
  - collect_drink: matched 25/74 (33.8%), others 19/224 (8.5%)
  - collect_sapling: matched 21/74 (28.4%), others 41/224 (18.3%)
  - place_table: matched 3/74 (4.1%), others 0/224 (0.0%)
  - eat_cow: matched 2/74 (2.7%), others 0/224 (0.0%)
  - place_plant: matched 3/74 (4.1%), others 5/224 (2.2%)
  - defeat_skeleton: matched 2/74 (2.7%), others 2/224 (0.9%)
  - defeat_zombie: matched 0/74 (0.0%), others 2/224 (0.9%)
\nDone. See ft_data/evals_filtered.jsonl and v3 DB for deeper analysis.
joshuapurtell@Mac synth-ai % 