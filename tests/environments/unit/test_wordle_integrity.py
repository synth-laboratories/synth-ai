import pytest

from synth_ai.environments.examples.wordle import create_wordle_taskset, WordleEnvironment


pytestmark = pytest.mark.asyncio


async def test_wordle_enforce_wordlist_rejects_unknown_word_consumes_turn():
    ts = await create_wordle_taskset(
        sample_size=1,
        word_length=5,
        max_guesses=4,
        enforce_wordlist=True,
        consume_invalid_attempts=True,
    )
    env = WordleEnvironment(ts.instances[0])
    await env.initialize()

    # Capture remaining guesses before invalid guess
    _, pub_before = env.engine.get_current_states_for_observation()
    rem_before = pub_before.remaining_guesses

    # 5-letter string that is very unlikely to be in base_word_list
    obs = await env.step({"guess": "zzzzz"})

    # Invalid word: penalty applied and turn consumed (since consume_invalid_attempts=True)
    assert obs["reward_last"] == -1.0
    assert obs["remaining_guesses"] == rem_before - 1


async def test_wordle_valid_guess_consumes_turn_and_updates_feedback():
    ts = await create_wordle_taskset(sample_size=1, word_length=5, max_guesses=4)
    env = WordleEnvironment(ts.instances[0])
    await env.initialize()

    # Ensure remaining guesses decrease by one on a valid-length guess
    _, pub_before = env.engine.get_current_states_for_observation()
    rem_before = pub_before.remaining_guesses

    obs = await env.step({"guess": "trace"})
    assert obs["remaining_guesses"] == rem_before - 1

    # Feedback string should match word length and use only G/Y/B
    assert len(obs["feedback"]) >= 1
    last_fb = obs["feedback"][-1]
    assert len(last_fb) == 5
    assert set(last_fb).issubset({"G", "Y", "B"})
