import pytest

from synth_ai.environments.examples.red.engine_helpers.reward_components import (
    BadgeVictoryReward,
    RouteExplorationReward,
    StrategicTrainingReward,
    BattleProgressionReward,
    GymPreparationReward,
    ItemCollectionReward,
    HealingManagementReward,
    EfficientExplorationReward,
    StepPenaltyComponent,
)


class TestComprehensiveRewardComponents:
    """Test the new comprehensive reward system"""

    @pytest.mark.asyncio
    async def test_badge_victory_reward(self):
        """Test badge victory reward - massive reward for main objective"""
        component = BadgeVictoryReward()

        # No badge change
        state = {"badges": 0x00}
        action = {"prev_badges": 0x00}
        reward = await component.score(state, action)
        assert reward == 0.0

        # First badge earned (Boulder Badge - main goal!)
        state = {"badges": 0x01}
        action = {"prev_badges": 0x00}
        reward = await component.score(state, action)
        assert reward == 50.0  # MASSIVE reward for completing objective

        # Already had badge
        state = {"badges": 0x01}
        action = {"prev_badges": 0x01}
        reward = await component.score(state, action)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_route_exploration_reward(self):
        """Test route exploration rewards - guides toward Pewter Gym"""
        component = RouteExplorationReward()

        # First visit to Route 1 (leaving Pallet Town)
        state = {"map_id": 1}
        action = {"prev_map_id": 0}
        reward = await component.score(state, action)
        assert reward == 2.0

        # Repeat visit to Route 1 (no reward)
        state = {"map_id": 1}
        action = {"prev_map_id": 2}
        reward = await component.score(state, action)
        assert reward == 0.0

        # Viridian Forest (challenging area)
        state = {"map_id": 5}
        action = {"prev_map_id": 4}
        reward = await component.score(state, action)
        assert reward == 2.0

        # Pewter City (target city)
        state = {"map_id": 6}
        action = {"prev_map_id": 1}
        reward = await component.score(state, action)
        assert reward == 1.5

        # Pewter Gym (GOAL AREA!)
        state = {"map_id": 7}
        action = {"prev_map_id": 6}
        reward = await component.score(state, action)
        assert reward == 5.0

    @pytest.mark.asyncio
    async def test_strategic_training_reward(self):
        """Test strategic training rewards"""
        component = StrategicTrainingReward()

        # Level 8 milestone (first time)
        state = {"party_level": 8}
        action = {"prev_party_level": 7}
        reward = await component.score(state, action)
        assert reward == 1.0  # milestone reward (level up separate)

        # Level 12 milestone (gym ready)
        state = {"party_level": 12}
        action = {"prev_party_level": 11}
        reward = await component.score(state, action)
        assert reward == 2.0  # milestone reward

        # Level 15 milestone (strong Pokemon)
        state = {"party_level": 15}
        action = {"prev_party_level": 14}
        reward = await component.score(state, action)
        assert reward == 3.0  # milestone reward

        # Regular level up (no milestone)
        state = {"party_level": 9}
        action = {"prev_party_level": 8}
        reward = await component.score(state, action)
        assert reward == 0.2

        # No level change
        state = {"party_level": 10}
        action = {"prev_party_level": 10}
        reward = await component.score(state, action)
        assert reward == 0.0

        # Test that milestones are only given once (but level ups can repeat)
        state = {"party_level": 13}
        action = {"prev_party_level": 12}
        reward = await component.score(state, action)
        assert reward == 0.2  # Level up reward, but no milestone since 13 not in milestones

    @pytest.mark.asyncio
    async def test_battle_progression_reward(self):
        """Test battle progression rewards"""
        component = BattleProgressionReward()

        # Enter battle (shows engagement)
        state = {"in_battle": True}
        action = {"prev_in_battle": False}
        reward = await component.score(state, action)
        assert reward == 0.1

        # Win battle (victory!)
        state = {"in_battle": False, "battle_outcome": 1}
        action = {"prev_in_battle": True}
        reward = await component.score(state, action)
        assert reward == 1.0

        # Lose battle (no reward)
        state = {"in_battle": False, "battle_outcome": 2}
        action = {"prev_in_battle": True}
        reward = await component.score(state, action)
        assert reward == 0.0

        # Not in battle transitions
        state = {"in_battle": False}
        action = {"prev_in_battle": False}
        reward = await component.score(state, action)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_gym_preparation_reward(self):
        """Test gym preparation reward"""
        component = GymPreparationReward()

        # Not in gym area
        state = {"map_id": 0, "party_level": 12, "party": [{"level": 12}]}
        action = {}
        reward = await component.score(state, action)
        assert reward == 0.0

        # In gym area but not prepared
        state = {"map_id": 6, "party_level": 5, "party": [{"level": 5}]}
        action = {}
        reward = await component.score(state, action)
        assert reward == 0.0

        # In gym area and prepared!
        state = {"map_id": 6, "party_level": 12, "party": [{"level": 12}]}
        action = {}
        reward = await component.score(state, action)
        assert reward == 3.0

        # Already rewarded
        state = {"map_id": 7, "party_level": 15, "party": [{"level": 15}]}
        action = {}
        reward = await component.score(state, action)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_item_collection_reward(self):
        """Test item collection rewards"""
        component = ItemCollectionReward()

        # Collect valuable item (Potion)
        state = {"inventory": [{"item_id": 1, "quantity": 1}]}
        action = {"prev_inventory": []}
        reward = await component.score(state, action)
        assert reward == 0.5

        # Collect multiple items
        state = {"inventory": [
            {"item_id": 1, "quantity": 1},
            {"item_id": 2, "quantity": 1},
            {"item_id": 50, "quantity": 1}
        ]}
        action = {"prev_inventory": [{"item_id": 1, "quantity": 1}]}
        reward = await component.score(state, action)
        assert reward == 0.5 + 0.1  # Pokeball (valuable) + random item

        # No new items
        state = {"inventory": [{"item_id": 1, "quantity": 1}]}
        action = {"prev_inventory": [{"item_id": 1, "quantity": 1}]}
        reward = await component.score(state, action)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_healing_management_reward(self):
        """Test healing management rewards"""
        component = HealingManagementReward()

        # Significant healing improvement
        state = {"party": [
            {"hp_percentage": 90.0},
            {"hp_percentage": 85.0}
        ]}
        action = {"prev_party": [
            {"hp_percentage": 50.0},
            {"hp_percentage": 45.0}
        ]}
        reward = await component.score(state, action)
        assert reward == 0.8

        # Health maintenance
        state = {"party": [
            {"hp_percentage": 95.0},
            {"hp_percentage": 90.0}
        ]}
        action = {"prev_party": [
            {"hp_percentage": 92.0},
            {"hp_percentage": 88.0}
        ]}
        reward = await component.score(state, action)
        assert reward == 0.05

        # Poor health
        state = {"party": [
            {"hp_percentage": 30.0}
        ]}
        action = {"prev_party": [
            {"hp_percentage": 25.0}
        ]}
        reward = await component.score(state, action)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_efficient_exploration_reward(self):
        """Test efficient exploration rewards"""
        component = EfficientExplorationReward()

        # New position discovered
        state = {"map_id": 0, "player_x": 10, "player_y": 10}
        action = {"prev_player_x": 9, "prev_player_y": 10}
        reward = await component.score(state, action)
        assert reward == 0.02

        # Repeat position (no reward)
        state = {"map_id": 0, "player_x": 10, "player_y": 10}
        action = {"prev_player_x": 10, "prev_player_y": 10}
        reward = await component.score(state, action)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_step_penalty_component(self):
        """Test step penalty (now disabled by default)"""
        component = StepPenaltyComponent()

        # Default penalty (now 0.0 - no penalty for existing)
        reward = await component.score({}, {})
        assert reward == 0.0

        # Custom penalty (can still be set if needed)
        component = StepPenaltyComponent(penalty=-0.01)
        reward = await component.score({}, {})
        assert reward == -0.01

    @pytest.mark.asyncio
    async def test_reward_determinism(self):
        """Test that rewards are deterministic and stateful"""
        component = RouteExplorationReward()

        # First visit to Route 1 should give reward
        state = {"map_id": 1}
        action = {"prev_map_id": 0}
        reward1 = await component.score(state, action)
        assert reward1 == 2.0

        # Second call with same inputs should be 0 (area already visited)
        reward2 = await component.score(state, action)
        assert reward2 == 0.0

        # Test with different component instance
        component2 = RouteExplorationReward()
        reward3 = await component2.score(state, action)
        assert reward3 == 2.0  # Fresh component gives reward again
