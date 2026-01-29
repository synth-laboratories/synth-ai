#!/usr/bin/env python3
"""
Reprex: tcg_py game_steps jumps from ~34 to max_steps instantly

ISSUE: When calling run_until_agent_turn() after P2's turn starts,
the game consumes ALL remaining steps (e.g., 34 -> 10000) in a single call,
even though only ~10 decision steps have occurred.

Expected: Game should progress turn-by-turn with reasonable step consumption.
Actual: AI opponent (P2) consumes thousands of steps in one run_until_agent_turn() call.
"""

import tcg_py

# Dragon Frontiers deck (same as localapi_ptcg.py)
DECK_1 = (
    ["df-061-ralts"] * 4
    + ["df-033-kirlia"] * 3
    + ["df-093-gardevoir-ex"] * 2
    + ["df-017-jynx"] * 2
    + ["df-070-vulpix"] * 3
    + ["df-008-ninetales"] * 2
    + ["df-010-snorlax"] * 2
    + ["df-082-tv-reporter"] * 4
    + ["df-079-prof-elms-training"] * 4
    + ["df-072-buffer-piece"] * 2
    + ["ENERGY-PSYCHIC"] * 16
    + ["ENERGY-FIRE"] * 16
)

DECK_2 = (
    ["df-068-trapinch"] * 4
    + ["df-024-vibrava"] * 3
    + ["df-092-flygon-ex"] * 2
    + ["df-009-pinsir"] * 3
    + ["df-003-heracross"] * 2
    + ["df-082-tv-reporter"] * 4
    + ["df-079-prof-elms-training"] * 3
    + ["df-072-buffer-piece"] * 2
    + ["ENERGY-PSYCHIC"] * 20
    + ["ENERGY-GRASS"] * 17
)


def main():
    max_steps = 10_000

    game = tcg_py.PtcgGame(
        p1_deck=DECK_1,
        p2_deck=DECK_2,
        game_seed=42,
        ai_seed=42,
        max_steps=max_steps,
    )

    decision_steps = 0

    print(f"Starting game with max_steps={max_steps}")
    print("-" * 60)

    prompt_detected = False
    while not game.is_game_over():
        # Get observation - this is where the bug occurs
        obs = game.run_until_agent_turn()

        if game.is_game_over():
            result = game.get_result()
            print("\nGAME OVER:")
            print(f"  decision_steps={decision_steps}")
            print(f"  game_steps={result.steps}")
            print(f"  end_reason={result.end_reason}")
            print(f"  winner={result.winner}")
            break

        result = game.get_result()
        game_steps = result.steps

        print(
            f"Decision {decision_steps}: player={obs.current_player}, phase={obs.phase}, "
            f"game_steps={game_steps}, actions={obs.available_actions}, has_prompt={obs.has_prompt}"
        )

        if obs.has_prompt:
            print("  -> Prompt detected for P1, stopping to verify prompt surface")
            prompt_detected = True
            break

        # If no actions, just step once
        if not obs.available_actions and not obs.has_prompt:
            print("  -> No actions, calling game.step()")
            game.step()
            decision_steps += 1
            continue

        # Auto-end turn if only EndTurn available
        if obs.available_actions == ["EndTurn"]:
            print("  -> Auto EndTurn")
            game.submit_action('{"action": "EndTurn"}')
            game.step()
            decision_steps += 1
            continue

        # Pick first available action (simplified for reprex)
        actions = obs.available_actions or []
        if "AttachEnergy" in str(actions):
            # Try to attach energy
            action = '{"action": "AttachEnergy", "energy_id": 41, "target_id": 4}'
        elif "DeclareAttack" in str(actions):
            action = '{"action": "DeclareAttack", "attack": "Hypnosis"}'
        elif "ChooseActive" in str(actions):
            action = '{"action": "ChooseActive", "card_id": 4}'
        elif "ChooseBench" in str(actions):
            action = '{"action": "ChooseBench", "card_ids": [11]}'
        elif "PlayBasic" in str(actions):
            action = '{"action": "PlayBasic", "card_id": 11}'
        elif "EndTurn" in actions:
            action = '{"action": "EndTurn"}'
        else:
            print(f"  -> Unknown actions: {actions}, stepping")
            game.step()
            decision_steps += 1
            continue

        print(f"  -> Submitting: {action[:50]}...")
        try:
            game.submit_action(action)
            game.step()
        except Exception as e:
            print(f"  -> Action failed: {e}, stepping anyway")
            game.step()

        decision_steps += 1

        # Safety break
        if decision_steps > 100:
            print("Safety break at 100 decision steps")
            break

    print("-" * 60)
    if prompt_detected:
        print("\nPROMPT SURFACE OK:")
        print(f"  - decision_steps={decision_steps}")
        print("  - prompt detected for P1 during setup")
    else:
        print("\nBUG REPRODUCED:")
        print(f"  - decision_steps={decision_steps} (expected: many)")
        print("  - game_steps hit max immediately")
        print(f"\nThe FIRST call to run_until_agent_turn() consumes ALL {max_steps} steps")
        print("without ever returning an observation to the agent (P1).")
        print("\nExpected behavior: run_until_agent_turn() should return whenever")
        print("P1 (the agent) needs to make a decision, not consume all steps internally.")


if __name__ == "__main__":
    main()
