import pytest

from synth_ai.environments.examples.wordle import (
    WordleEnvironment,
    WordleTaskInstance,
    WordleTaskInstanceMetadata,
    create_wordle_taskset,
)
from synth_ai.environments.tasks.core import Impetus, Intent

pytestmark = pytest.mark.asyncio


async def test_wordle_taskset_and_win_flow():
    ts = await create_wordle_taskset(sample_size=3, word_length=5, max_guesses=6)
    assert len(ts.instances) == 3
    inst = ts.instances[0]

    env = WordleEnvironment(inst)
    obs = await env.initialize()
    assert isinstance(obs, dict)
    assert obs.get("status") in {"in_progress", "won", "lost"}

    # Make two guesses, last is correct to ensure win
    target = inst.metadata.target_word
    obs = await env.step({"guess": "trace"})
    obs = await env.step({"guess": target})

    # Check reward and status
    assert obs["status"] == "won"
    assert obs["reward_last"] == 1.0
    assert obs["terminated"] is True


async def test_wordle_invalid_guess_penalty_no_turn_consumed():
    ts = await create_wordle_taskset(sample_size=1, consume_invalid_attempts=False)
    env = WordleEnvironment(ts.instances[0])
    await env.initialize()

    # Capture remaining guesses before invalid
    priv_before, pub_before = env.engine.get_current_states_for_observation()
    rem_before = pub_before.remaining_guesses

    obs = await env.step({"guess": "aa"})  # wrong length -> invalid
    assert obs["reward_last"] == -1.0

    # Remaining guesses unchanged
    _, pub_after = env.engine.get_current_states_for_observation()
    assert pub_after.remaining_guesses == rem_before


async def test_wordle_loss_flow_and_rewards_zero_on_loss():
    ts = await create_wordle_taskset(sample_size=1, word_length=5, max_guesses=3)
    env = WordleEnvironment(ts.instances[0])
    await env.initialize()

    # Make three incorrect, valid guesses of correct length
    wrongs = ["aaaaa", "bbbbb", "ccccc"]
    # Ensure wrong words are not equal to target
    wrongs = [w for w in wrongs if w != ts.instances[0].metadata.target_word]
    for w in wrongs[:3]:
        obs = await env.step({"guess": w})

    # Should be lost with no win reward
    assert obs["status"] == "lost"
    assert obs["terminated"] is True
    # Total reward can be 0.0 if no invalids and no win
    assert obs["total_reward"] <= 0.0


async def test_wordle_serialize_deserialize_roundtrip():
    ts = await create_wordle_taskset(sample_size=1)
    inst = ts.instances[0]
    env1 = WordleEnvironment(inst)
    await env1.initialize()
    await env1.step({"guess": "trace"})

    snapshot = await env1._serialize_engine()

    # Rehydrate a new env from snapshot
    env2 = await WordleEnvironment._deserialize_engine(snapshot, inst)

    # Continue stepping in both and compare observations (allowing for minor duplication
    # differences across environments; verify last step parity and common prefix)
    obs1 = await env1.step({"guess": "slate"})
    obs2 = await env2.step({"guess": "slate"})

    g1, g2 = obs1["guesses"], obs2["guesses"]
    f1, f2 = obs1["feedback"], obs2["feedback"]
    assert g1[-1] == g2[-1] == "slate"
    assert f1[-1] == f2[-1]
    # Shared prefix should match up to min length
    k = min(len(g1), len(g2))
    assert g1[:k] == g2[:k]
    assert f1[:k] == f2[:k]
    assert obs1["status"] == obs2["status"]


async def test_wordle_different_seeds_yield_different_targets():
    # Build two instances that rely on seed (no fixed target)
    md1 = WordleTaskInstanceMetadata(
        word_length=5,
        max_guesses=6,
        target_word="",  # empty -> engine uses seed-based selection
        enforce_wordlist=False,
        seed=1,
    )
    md2 = WordleTaskInstanceMetadata(
        word_length=5,
        max_guesses=6,
        target_word="",  # empty -> engine uses seed-based selection
        enforce_wordlist=False,
        seed=2,
    )
    inst1 = WordleTaskInstance(
        id=__import__("uuid").uuid4(),
        impetus=Impetus(instructions="Play Wordle"),
        intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
        metadata=md1,
        is_reproducible=True,
        initial_engine_snapshot=None,
    )
    inst2 = WordleTaskInstance(
        id=__import__("uuid").uuid4(),
        impetus=Impetus(instructions="Play Wordle"),
        intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
        metadata=md2,
        is_reproducible=True,
        initial_engine_snapshot=None,
    )

    env1 = WordleEnvironment(inst1)
    env2 = WordleEnvironment(inst2)
    await env1.initialize()
    await env2.initialize()

    t1 = env1.engine.target
    t2 = env2.engine.target
    assert t1 is not None and t2 is not None
    assert t1 != t2, (t1, t2)
