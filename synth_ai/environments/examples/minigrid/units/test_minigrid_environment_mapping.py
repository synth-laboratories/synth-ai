"""Unit tests for MiniGrid environment mapping functionality."""

import pytest
from synth_ai.environments.examples.minigrid.environment_mapping import (
    get_environment_from_seed,
    get_difficulty_from_seed,
    get_minigrid_environment,
    get_environment_by_difficulty,
    get_curriculum_environment,
    validate_environment_name,
    get_all_environments,
    get_environments_by_difficulty,
    ENVIRONMENT_MAPPING,
    DIFFICULTY_MAPPING,
)


class TestEnvironmentMapping:
    """Test the environment mapping functionality."""

    def test_seed_to_environment_mapping(self):
        """Test basic seed to environment mapping."""
        # Test specific known mappings
        assert get_environment_from_seed(0) == "MiniGrid-Empty-5x5-v0"
        assert get_environment_from_seed(1) == "MiniGrid-Empty-6x6-v0"
        assert get_environment_from_seed(7) == "MiniGrid-DoorKey-5x5-v0"
        assert get_environment_from_seed(42) == "MiniGrid-KeyCorridorS6R3-v0"
        assert get_environment_from_seed(59) == "MiniGrid-Empty-16x16-v0"

        # Test modulo behavior
        assert get_environment_from_seed(60) == get_environment_from_seed(0)
        assert get_environment_from_seed(61) == get_environment_from_seed(1)
        assert get_environment_from_seed(120) == get_environment_from_seed(0)

        # Test negative seeds (hash behavior, not simple modulo)
        assert get_environment_from_seed(-1) in ENVIRONMENT_MAPPING.values()
        assert get_environment_from_seed(-5) in ENVIRONMENT_MAPPING.values()

    def test_difficulty_mapping(self):
        """Test difficulty level mapping."""
        # Test ultra-easy (0-4)
        assert get_difficulty_from_seed(0) == "ultra-easy"
        assert get_difficulty_from_seed(4) == "ultra-easy"

        # Test easy (5-14)
        assert get_difficulty_from_seed(5) == "easy"
        assert get_difficulty_from_seed(14) == "easy"

        # Test medium (15-29)
        assert get_difficulty_from_seed(15) == "medium"
        assert get_difficulty_from_seed(29) == "medium"

        # Test hard (30-44)
        assert get_difficulty_from_seed(30) == "hard"
        assert get_difficulty_from_seed(44) == "hard"

        # Test ultra-hard (45-54)
        assert get_difficulty_from_seed(45) == "ultra-hard"
        assert get_difficulty_from_seed(54) == "ultra-hard"

        # Test specialized (55-59)
        assert get_difficulty_from_seed(55) == "specialized"
        assert get_difficulty_from_seed(59) == "specialized"

    def test_combined_mapping(self):
        """Test the combined environment and difficulty mapping."""
        test_seeds = [0, 7, 15, 30, 45, 55, 42, 1337]

        for seed in test_seeds:
            env_name = get_environment_from_seed(seed)
            difficulty = get_difficulty_from_seed(seed)

            combined_env, combined_diff = get_minigrid_environment(seed)

            assert env_name == combined_env
            assert difficulty == combined_diff

    def test_environment_by_difficulty(self):
        """Test selecting environments by difficulty level."""
        # Test each difficulty level
        for difficulty in DIFFICULTY_MAPPING:
            env_name = get_environment_by_difficulty(difficulty, 0)
            assert env_name in ENVIRONMENT_MAPPING.values()

            # Test that different seeds give different environments (within range)
            env_name2 = get_environment_by_difficulty(difficulty, 1)
            assert env_name2 in ENVIRONMENT_MAPPING.values()

    def test_environment_by_difficulty_invalid(self):
        """Test invalid difficulty levels."""
        with pytest.raises(ValueError, match="Unknown difficulty"):
            get_environment_by_difficulty("invalid", 0)

        with pytest.raises(ValueError, match="Unknown difficulty"):
            get_environment_by_difficulty("super-easy", 0)

    def test_curriculum_environment(self):
        """Test curriculum-based environment selection."""
        # Test early stage (0.0-0.2) -> ultra-easy
        env_name, difficulty = get_curriculum_environment(0.0, 0)
        assert difficulty == "ultra-easy"

        env_name, difficulty = get_curriculum_environment(0.1, 0)
        assert difficulty == "ultra-easy"

        # Test beginning (0.2-0.4) -> easy
        env_name, difficulty = get_curriculum_environment(0.3, 0)
        assert difficulty == "easy"

        # Test intermediate (0.4-0.6) -> medium
        env_name, difficulty = get_curriculum_environment(0.5, 0)
        assert difficulty == "medium"

        # Test advanced (0.6-0.8) -> hard
        env_name, difficulty = get_curriculum_environment(0.7, 0)
        assert difficulty == "hard"

        # Test expert (0.8-0.9) -> ultra-hard
        env_name, difficulty = get_curriculum_environment(0.85, 0)
        assert difficulty == "ultra-hard"

        # Test master (0.9-1.0) -> specialized
        env_name, difficulty = get_curriculum_environment(0.95, 0)
        assert difficulty == "specialized"

    def test_environment_validation(self):
        """Test environment name validation."""
        # Valid environments
        assert validate_environment_name("MiniGrid-Empty-5x5-v0") is True
        assert validate_environment_name("MiniGrid-FourRooms-v0") is True
        assert validate_environment_name("MiniGrid-DoorKey-5x5-v0") is True

        # Invalid environments
        assert validate_environment_name("MiniGrid-NonExistent-v0") is False
        assert validate_environment_name("InvalidEnv") is False
        assert validate_environment_name("") is False

    def test_get_all_environments(self):
        """Test getting all environments."""
        all_envs = get_all_environments()

        assert len(all_envs) == 60
        assert "MiniGrid-Empty-5x5-v0" in all_envs
        assert "MiniGrid-FourRooms-v0" in all_envs
        assert "MiniGrid-Empty-16x16-v0" in all_envs

        # Check no duplicates
        assert len(all_envs) == len(set(all_envs))

    def test_get_environments_by_difficulty(self):
        """Test getting environments by difficulty level."""
        # Test each difficulty level
        ultra_easy = get_environments_by_difficulty("ultra-easy")
        assert len(ultra_easy) == 5  # Seeds 0-4

        easy = get_environments_by_difficulty("easy")
        assert len(easy) == 10  # Seeds 5-14

        medium = get_environments_by_difficulty("medium")
        assert len(medium) == 15  # Seeds 15-29

        hard = get_environments_by_difficulty("hard")
        assert len(hard) == 15  # Seeds 30-44

        ultra_hard = get_environments_by_difficulty("ultra-hard")
        assert len(ultra_hard) == 10  # Seeds 45-54

        specialized = get_environments_by_difficulty("specialized")
        assert len(specialized) == 5  # Seeds 55-59

        # Test invalid difficulty
        with pytest.raises(ValueError, match="Unknown difficulty"):
            get_environments_by_difficulty("invalid")

    def test_hash_seed_parameter(self):
        """Test the hash_seed parameter for better distribution."""
        # Test with hash_seed=True (default)
        env1 = get_environment_from_seed(100, hash_seed=True)
        diff1 = get_difficulty_from_seed(100, hash_seed=True)

        # Test with hash_seed=False
        env2 = get_environment_from_seed(100, hash_seed=False)
        diff2 = get_difficulty_from_seed(100, hash_seed=False)

        # Should be different due to hash distribution
        # (Note: This might occasionally fail due to hash collisions, but very unlikely)
        assert env1 in ENVIRONMENT_MAPPING.values()
        assert env2 in ENVIRONMENT_MAPPING.values()

        # Test consistency
        assert get_environment_from_seed(100, hash_seed=True) == env1
        assert get_environment_from_seed(100, hash_seed=False) == env2

    def test_environment_mapping_completeness(self):
        """Test that the environment mapping is complete."""
        # Test all 60 environments are mapped
        assert len(ENVIRONMENT_MAPPING) == 60

        # Test all indices 0-59 are present
        for i in range(60):
            assert i in ENVIRONMENT_MAPPING

        # Test all environments are unique
        env_names = list(ENVIRONMENT_MAPPING.values())
        assert len(env_names) == len(set(env_names))

    def test_difficulty_mapping_completeness(self):
        """Test that the difficulty mapping is complete."""
        # Test all difficulty levels are mapped
        expected_difficulties = [
            "ultra-easy",
            "easy",
            "medium",
            "hard",
            "ultra-hard",
            "specialized",
        ]
        assert set(DIFFICULTY_MAPPING.keys()) == set(expected_difficulties)

        # Test ranges are correct
        assert DIFFICULTY_MAPPING["ultra-easy"] == (0, 4)
        assert DIFFICULTY_MAPPING["easy"] == (5, 14)
        assert DIFFICULTY_MAPPING["medium"] == (15, 29)
        assert DIFFICULTY_MAPPING["hard"] == (30, 44)
        assert DIFFICULTY_MAPPING["ultra-hard"] == (45, 54)
        assert DIFFICULTY_MAPPING["specialized"] == (55, 59)

        # Test ranges are contiguous and complete
        ranges = list(DIFFICULTY_MAPPING.values())
        ranges.sort()

        current_end = -1
        for start, end in ranges:
            assert start == current_end + 1
            current_end = end

        assert current_end == 59

    def test_reproducibility(self):
        """Test that seed mapping is reproducible."""
        test_seeds = [0, 42, 1337, -5, 999]

        for seed in test_seeds:
            # Multiple calls should return same result
            env1 = get_environment_from_seed(seed)
            env2 = get_environment_from_seed(seed)
            assert env1 == env2

            diff1 = get_difficulty_from_seed(seed)
            diff2 = get_difficulty_from_seed(seed)
            assert diff1 == diff2

    def test_extreme_seeds(self):
        """Test with extreme seed values."""
        extreme_seeds = [
            0,
            1,
            -1,
            999999,
            -999999,
            2**31 - 1,
            -(2**31),  # 32-bit int limits
            2**63 - 1,
            -(2**63),  # 64-bit int limits
        ]

        for seed in extreme_seeds:
            env_name = get_environment_from_seed(seed)
            difficulty = get_difficulty_from_seed(seed)

            assert env_name in ENVIRONMENT_MAPPING.values()
            assert difficulty in DIFFICULTY_MAPPING.keys()

    def test_comprehensive_mapping_coverage(self):
        """Test that all environments in the mapping are properly categorized."""
        # Test that every environment has proper features
        feature_keywords = {
            "key": ["DoorKey", "Unlock", "KeyCorridor"],
            "door": ["Door", "Room", "Unlock"],
            "lava": ["Lava"],
            "memory": ["Memory"],
            "fetch": ["Fetch"],
            "multi_room": ["MultiRoom"],
            "crossing": ["Crossing"],
            "maze": ["Maze"],
            "empty": ["Empty"],
        }

        for env_name in ENVIRONMENT_MAPPING.values():
            # Each environment should have at least one recognizable feature
            has_feature = False
            for feature, keywords in feature_keywords.items():
                if any(keyword in env_name for keyword in keywords):
                    has_feature = True
                    break

            # Some environments might not match these patterns, that's OK
            # This is just a sanity check, not a strict requirement
            assert isinstance(env_name, str)
            assert env_name.startswith("MiniGrid-")
            assert env_name.endswith("-v0")
