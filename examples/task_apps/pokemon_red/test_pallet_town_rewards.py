"""
Test script for Pallet Town Progression Rewards

This script demonstrates the reward function by simulating
a sequence of states representing the ideal Pallet Town progression.
"""

import asyncio
from synth_ai.environments.examples.red.engine_helpers.reward_library.pallet_town_progression import (
    PalletTownProgressionCompositeReward,
)


async def main():
    """Simulate a perfect Pallet Town run and show rewards"""
    
    reward_fn = PalletTownProgressionCompositeReward()
    total_reward = 0.0
    
    print("=" * 70)
    print("PALLET TOWN PROGRESSION - REWARD SIMULATION")
    print("=" * 70)
    print()
    
    # Step 1: Start in bedroom (Map 1)
    state1 = {
        "map_id": 1,
        "player_x": 3,
        "player_y": 4,
        "party_count": 0,
        "in_battle": False,
        "text_box_active": False,
        "battle_outcome": 0,
        "enemy_hp_current": 0,
        "enemy_hp_max": 0,
        "enemy_hp_percentage": 0.0,
    }
    action1 = {
        "prev_map_id": 1,
        "prev_party_count": 0,
        "prev_in_battle": False,
        "prev_text_box_active": False,
        "prev_enemy_hp_current": 0,
        "prev_enemy_hp_percentage": 0.0,
    }
    
    # Step 2: Go downstairs (Map 1 -> Map 2)
    state2 = {**state1, "map_id": 2, "player_y": 8}
    action2 = {**action1, "prev_map_id": 1}
    
    r = await reward_fn.score(state2, action2)
    total_reward += r
    print(f"✓ Leave bedroom (Map 1→2):                    +{r:.0f} points")
    
    # Step 3: Exit house (Map 2 -> Map 0)
    state3 = {**state2, "map_id": 0, "player_x": 5, "player_y": 7}
    action3 = {**action2, "prev_map_id": 2}
    
    r = await reward_fn.score(state3, action3)
    total_reward += r
    print(f"✓ Exit house to Pallet Town (Map 2→0):        +{r:.0f} points")
    
    # Step 4: Navigate to and enter Oak's Lab (Map 0 -> Map 3)
    state4 = {**state3, "map_id": 3, "player_x": 4, "player_y": 11}
    action4 = {**action3, "prev_map_id": 0}
    
    r = await reward_fn.score(state4, action4)
    total_reward += r
    print(f"✓ Find and enter Oak's Lab (Map 0→3):         +{r:.0f} points")
    
    # Step 5: Talk to Oak (text box appears)
    state5 = {**state4, "text_box_active": True}
    action5 = {**action4, "prev_text_box_active": False}
    
    r = await reward_fn.score(state5, action5)
    total_reward += r
    print(f"✓ Talk to Professor Oak:                      +{r:.0f} points")
    
    # Step 6: Receive starter Pokemon (party count 0 -> 1)
    state6 = {
        **state5,
        "party_count": 1,
        "party_pokemon": [
            {
                "species_id": 4,  # Charmander
                "level": 5,
                "hp_current": 20,
                "hp_max": 20,
                "hp_percentage": 100.0,
            }
        ],
    }
    action6 = {**action5, "prev_party_count": 0}
    
    r = await reward_fn.score(state6, action6)
    total_reward += r
    print(f"✓ Receive starter Pokemon:                    +{r:.0f} points")
    
    # Step 7: Enter first battle
    state7 = {**state6, "in_battle": True, "text_box_active": False,
              "enemy_hp_current": 20, "enemy_hp_max": 20, "enemy_hp_percentage": 100.0}
    action7 = {**action6, "prev_in_battle": False, "prev_text_box_active": True}
    
    r = await reward_fn.score(state7, action7)
    total_reward += r
    print(f"✓ Enter first battle with rival:              +{r:.0f} points")
    
    # Step 8-12: Deal damage (5 attacks)
    print()
    print("Battle sequence:")
    for i in range(5):
        prev_hp = 20 - (i * 4)
        curr_hp = 20 - ((i + 1) * 4)
        state_dmg = {
            **state7,
            "enemy_hp_current": curr_hp,
            "enemy_hp_percentage": (curr_hp / 20) * 100,
        }
        action_dmg = {
            **action7,
            "prev_in_battle": True,
            "prev_enemy_hp_current": prev_hp,
            "prev_enemy_hp_percentage": (prev_hp / 20) * 100,
        }
        
        r = await reward_fn.score(state_dmg, action_dmg)
        total_reward += r
        
        # Check for half HP and low HP milestones
        if r > 5:  # Got bonus reward
            if (prev_hp / 20) >= 0.5 and (curr_hp / 20) < 0.5:
                print(f"  → Attack {i+1}: Enemy HP {prev_hp}→{curr_hp} (+5) + Half HP bonus (+25) = +{r:.0f}")
            elif (prev_hp / 20) >= 0.25 and (curr_hp / 20) < 0.25:
                print(f"  → Attack {i+1}: Enemy HP {prev_hp}→{curr_hp} (+5) + Low HP bonus (+35) = +{r:.0f}")
        else:
            print(f"  → Attack {i+1}: Enemy HP {prev_hp}→{curr_hp}              +{r:.0f} points")
    
    print()
    
    # Step 13: Win battle
    state13 = {
        **state7,
        "in_battle": False,
        "battle_outcome": 1,  # Win
        "enemy_hp_current": 0,
        "enemy_hp_percentage": 0.0,
        "battle_turn": 4,
        "party_pokemon": [
            {
                "species_id": 4,
                "level": 5,
                "hp_current": 15,  # 75% HP
                "hp_max": 20,
                "hp_percentage": 75.0,
            }
        ],
    }
    action13 = {
        **action7,
        "prev_in_battle": True,
        "prev_enemy_hp_current": 0,
    }
    
    r = await reward_fn.score(state13, action13)
    total_reward += r
    print(f"✓ Win first battle:                            +{r:.0f} points")
    
    # Step 14: Exit lab with Pokemon (Map 3 -> Map 0)
    state14 = {**state13, "map_id": 0, "player_x": 5, "player_y": 11}
    action14 = {**action13, "prev_map_id": 3}
    
    r = await reward_fn.score(state14, action14)
    total_reward += r
    print(f"✓ Exit Oak's Lab with Pokemon (Map 3→0):      +{r:.0f} points")
    
    print()
    print("=" * 70)
    print(f"TOTAL REWARD: {total_reward:.0f} points")
    print("=" * 70)
    print()
    print("Breakdown by category:")
    print("  Navigation:      150 points (bedroom, house, lab, exit)")
    print("  Story:           150 points (talk to Oak, get Pokemon)")
    print("  Battle:          335 points (enter, damage, milestones, win)")
    print("  Efficiency:      ~100 points (battle speed, health, navigation)")
    print()


if __name__ == "__main__":
    asyncio.run(main())



