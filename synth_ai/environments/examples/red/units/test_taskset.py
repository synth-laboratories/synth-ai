import uuid
from pathlib import Path
from synth_ai.environments.examples.red.taskset import TASK, INSTANCE, PokemonRedTaskInstance
from synth_ai.environments.tasks.core import (
    Task,
    TaskInstance,
    Impetus,
    Intent,
    TaskInstanceMetadata,
)


class TestPokemonRedTaskset:
    """Test Pokemon Red task definitions"""

    def test_task_structure(self):
        """Test main task structure"""
        assert isinstance(TASK, Task)
        assert "Pokemon Red" in TASK.global_premises
        assert "Pewter" in TASK.global_premises
        assert "Pikachu" in TASK.global_premises
        assert "glitches" in TASK.global_constraints.lower()
        assert "Brock" in TASK.global_objectives
        assert "Boulder Badge" in TASK.global_objectives
        assert isinstance(TASK.shared_env_params, dict)

    def test_task_instance_structure(self):
        """Test task instance structure"""
        assert isinstance(INSTANCE, PokemonRedTaskInstance)
        assert isinstance(INSTANCE, TaskInstance)
        assert str(INSTANCE.id) == "12345678-1234-5678-9abc-123456789abc"
        assert isinstance(INSTANCE.impetus, Impetus)
        assert isinstance(INSTANCE.intent, Intent)
        assert INSTANCE.is_reproducible is True

    def test_task_instance_impetus(self):
        """Test task instance impetus"""
        impetus = INSTANCE.impetus
        assert "Pewter Gym" in impetus.instructions
        assert "Brock" in impetus.instructions
        assert "Boulder Badge" in impetus.instructions

    def test_task_instance_intent(self):
        """Test task instance intent"""
        intent = INSTANCE.intent
        assert "Boulder Badge" in intent.rubric
        assert "Brock" in intent.rubric
        assert "Pewter Gym" in intent.rubric

    def test_task_instance_metadata(self):
        """Test task instance metadata"""
        metadata = INSTANCE.metadata
        assert isinstance(metadata, TaskInstanceMetadata)
        # TaskInstanceMetadata is a simple dataclass with no required fields currently

    def test_initial_engine_snapshot(self):
        """Test initial engine snapshot configuration"""
        # Test that snapshot path is properly configured
        if INSTANCE.initial_engine_snapshot:
            assert isinstance(INSTANCE.initial_engine_snapshot, Path)
            assert INSTANCE.initial_engine_snapshot.name == "pewter_start.state"
            assert "snapshots" in str(INSTANCE.initial_engine_snapshot)
        else:
            # Snapshot file doesn't exist, which is expected in test environment
            expected_path = Path(__file__).parent.parent / "snapshots" / "pewter_start.state"
            assert not expected_path.exists()

    def test_pokemon_red_task_instance_type(self):
        """Test PokemonRedTaskInstance class"""
        assert issubclass(PokemonRedTaskInstance, TaskInstance)

        # Test that we can create instances
        custom_instance = PokemonRedTaskInstance(
            id=uuid.uuid4(),
            impetus=Impetus(instructions="Test instructions"),
            intent=Intent(
                rubric="Test goal: achieve something",
                gold_trajectories=None,
                gold_state_diff={},
            ),
            metadata=TaskInstanceMetadata(),
            is_reproducible=False,
            initial_engine_snapshot=None,
        )

        assert isinstance(custom_instance.id, uuid.UUID)
        assert custom_instance.is_reproducible is False
        assert custom_instance.initial_engine_snapshot is None

    def test_task_fields_not_empty(self):
        """Test that important task fields are not empty"""
        assert len(TASK.global_premises.strip()) > 0
        assert len(TASK.global_constraints.strip()) > 0
        assert len(TASK.global_objectives.strip()) > 0
        assert len(INSTANCE.impetus.instructions.strip()) > 0

    def test_task_consistency(self):
        """Test consistency between task and instance"""
        # Both should mention similar concepts
        task_text = f"{TASK.global_premises} {TASK.global_objectives}".lower()
        instance_text = INSTANCE.impetus.instructions.lower()

        # Key concepts should appear in both
        key_concepts = ["brock", "pewter", "badge"]
        for concept in key_concepts:
            assert concept in task_text, f"Concept '{concept}' missing from task"
            assert concept in instance_text, f"Concept '{concept}' missing from instance"

    def test_snapshot_path_structure(self):
        """Test snapshot path structure"""
        expected_path = Path(__file__).parent.parent / "snapshots" / "pewter_start.state"

        # The path should be structured correctly even if file doesn't exist
        assert expected_path.parent.name == "snapshots"
        assert expected_path.name == "pewter_start.state"
        assert expected_path.suffix == ".state"
