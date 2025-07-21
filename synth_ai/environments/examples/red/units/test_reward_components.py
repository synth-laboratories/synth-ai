import pytest
from synth_ai.environments.examples.red.engine_helpers.reward_components import (
    BadgeRewardComponent,
    MapTransitionComponent,
    BattleVictoryComponent,
    LevelUpComponent,
    XPGainComponent,
    StepPenaltyComponent,
    MenuPenaltyComponent,
)


class TestRewardComponents:
    """Test reward component calculations"""

    @pytest.mark.asyncio
    async def test_badge_reward_component(self):
        """Test badge reward calculation"""
        component = BadgeRewardComponent()

        # No new badges
        state = {"badges": 0x01}
        action = {"prev_badges": 0x01}
        reward = await component.score(state, action)
        assert reward == 0.0

        # One new badge
        state = {"badges": 0x03}  # Boulder + Cascade
        action = {"prev_badges": 0x01}  # Just Boulder
        reward = await component.score(state, action)
        assert reward == 1.0

        # Multiple new badges (unlikely but possible)
        state = {"badges": 0x07}  # First 3 badges
        action = {"prev_badges": 0x01}  # Just Boulder
        reward = await component.score(state, action)
        assert reward == 2.0

        # First badge ever
        state = {"badges": 0x01}
        action = {"prev_badges": 0x00}
        reward = await component.score(state, action)
        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_map_transition_component(self):
        """Test map transition reward"""
        component = MapTransitionComponent()

        # No map change
        state = {"map_id": 3}
        action = {"prev_map_id": 3}
        reward = await component.score(state, action)
        assert reward == 0.0

        # Map changed
        state = {"map_id": 4}
        action = {"prev_map_id": 3}
        reward = await component.score(state, action)
        assert reward == 0.1

        # No previous map (first step)
        state = {"map_id": 3}
        action = {}
        reward = await component.score(state, action)
        assert reward == 0.1  # Default prev_map is -1

    @pytest.mark.asyncio
    async def test_battle_victory_component(self):
        """Test battle victory reward"""
        component = BattleVictoryComponent()

        # Not transitioning from battle
        state = {"in_battle": False, "battle_outcome": 1}
        action = {"prev_in_battle": False}
        reward = await component.score(state, action)
        assert reward == 0.0

        # Still in battle
        state = {"in_battle": True, "battle_outcome": 0}
        action = {"prev_in_battle": True}
        reward = await component.score(state, action)
        assert reward == 0.0

        # Won battle (transitioned from battle to not battle with victory)
        state = {"in_battle": False, "battle_outcome": 1}
        action = {"prev_in_battle": True}
        reward = await component.score(state, action)
        assert reward == 0.5

        # Lost battle
        state = {"in_battle": False, "battle_outcome": 2}
        action = {"prev_in_battle": True}
        reward = await component.score(state, action)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_level_up_component(self):
        """Test level up reward"""
        component = LevelUpComponent()

        # No level change
        state = {"party_level": 10}
        action = {"prev_party_level": 10}
        reward = await component.score(state, action)
        assert reward == 0.0

        # Level up by 1
        state = {"party_level": 11}
        action = {"prev_party_level": 10}
        reward = await component.score(state, action)
        assert reward == 0.3

        # Level up by multiple (rare candy usage)
        state = {"party_level": 13}
        action = {"prev_party_level": 10}
        reward = await component.score(state, action)
        assert reward == pytest.approx(0.9)  # 3 levels * 0.3

        # Level decreased (shouldn't happen, but test bounds)
        state = {"party_level": 8}
        action = {"prev_party_level": 10}
        reward = await component.score(state, action)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_xp_gain_component(self):
        """Test XP gain reward"""
        component = XPGainComponent()

        # No XP change
        state = {"party_xp": 1000}
        action = {"prev_party_xp": 1000}
        reward = await component.score(state, action)
        assert reward == 0.0

        # XP gained
        state = {"party_xp": 1500}
        action = {"prev_party_xp": 1000}
        reward = await component.score(state, action)
        assert reward == 0.5  # 500 * 0.001

        # XP decreased (shouldn't happen)
        state = {"party_xp": 800}
        action = {"prev_party_xp": 1000}
        reward = await component.score(state, action)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_step_penalty_component(self):
        """Test step penalty"""
        component = StepPenaltyComponent()

        # Default penalty
        reward = await component.score({}, {})
        assert reward == -0.001

        # Custom penalty
        component = StepPenaltyComponent(penalty=-0.01)
        reward = await component.score({}, {})
        assert reward == -0.01

    @pytest.mark.asyncio
    async def test_menu_penalty_component(self):
        """Test menu penalty (currently no-op)"""
        component = MenuPenaltyComponent()

        reward = await component.score({}, {})
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        badge_component = BadgeRewardComponent()

        # Missing prev_badges key
        state = {"badges": 0x01}
        action = {}
        reward = await badge_component.score(state, action)
        assert reward == 1.0  # Default prev_badges is 0

        # All badges at once (impossible but test)
        state = {"badges": 0xFF}
        action = {"prev_badges": 0x00}
        reward = await badge_component.score(state, action)
        assert reward == 8.0
