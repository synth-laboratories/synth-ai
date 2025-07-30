import pytest
from unittest.mock import Mock, patch, AsyncMock

# Add imports for the new dataclasses
from synth_ai.environments.examples.red.engine import (
    GameWorldState,
    PlayerProgressState,
    GameSystemState,
    PokemonData,
)
from synth_ai.environments.examples.red.environment import (
    PokemonRedEnvironment,
    PokemonRedPublicState,
    PokemonRedPrivateState,
    PressButtonTool,
    PokemonRedObservationCallable,
)
from synth_ai.environments.environment.tools import EnvToolCall, ToolResult
from synth_ai.environments.examples.red.taskset import INSTANCE as DEFAULT_TASK


class TestPokemonRedEnvironment:
    """Test Pokemon Red environment wrapper"""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine"""
        engine = Mock()
        engine._reset_engine = AsyncMock(
            return_value=(
                PokemonRedPrivateState(
                    reward_last_step=0.0,
                    total_reward=0.0,
                    terminated=False,
                    truncated=False,
                    step_count=0,
                ),
                create_test_public_state(
                    map_id=3,
                    player_x=10,
                    player_y=8,
                    badges=0,
                    in_battle=False,
                    party_level=10,
                    party_hp_current=35,
                    party_hp_max=35,
                    party_xp=1000,
                    step_count=0,
                ),
            )
        )
        engine._step_engine = AsyncMock()
        engine._serialize_engine = AsyncMock()
        engine._create_states = Mock()
        return engine

    @patch("src.examples.red.environment.PokemonRedEngine")
    def test_environment_initialization(self, mock_engine_class, mock_engine):
        """Test environment initialization"""
        mock_engine_class.return_value = mock_engine

        env = PokemonRedEnvironment()

        assert env.name == "PokemonRed"
        assert env.task_instance == DEFAULT_TASK
        assert env.engine == mock_engine
        assert isinstance(env._press_button_tool, PressButtonTool)

    @patch("src.examples.red.environment.PokemonRedEngine")
    @pytest.mark.asyncio
    async def test_initialize(self, mock_engine_class, mock_engine):
        """Test environment initialization"""
        mock_engine_class.return_value = mock_engine

        env = PokemonRedEnvironment()
        obs = await env.initialize()

        mock_engine._reset_engine.assert_called_once()
        assert "position" in obs
        assert "badges_earned" in obs
        assert obs["badges_earned"] == 0
        assert obs["party_level"] == 10

    @patch("src.examples.red.environment.PokemonRedEngine")
    @pytest.mark.asyncio
    async def test_terminate(self, mock_engine_class, mock_engine):
        """Test environment termination"""
        mock_engine_class.return_value = mock_engine
        mock_engine._create_states.return_value = (
            PokemonRedPrivateState(
                reward_last_step=0.0,
                total_reward=10.5,
                terminated=True,
                truncated=False,
                step_count=42,
            ),
            create_test_public_state(
                map_id=3,
                player_x=10,
                player_y=8,
                badges=1,
                in_battle=False,
                party_level=12,
                party_hp_current=30,
                party_hp_max=35,
                party_xp=1500,
                step_count=42,
            ),
        )

        env = PokemonRedEnvironment()
        obs = await env.terminate()

        assert obs["terminated"] is True
        assert "message" in obs

    def test_validate_tool_calls_single_call(self):
        """Test tool call validation with single call"""
        with patch("src.examples.red.environment.PokemonRedEngine"):
            env = PokemonRedEnvironment()

            call = EnvToolCall(tool="press_button", args={"button": "A"})
            validated = env.validate_tool_calls(call)

            assert validated == call

    def test_validate_tool_calls_list(self):
        """Test tool call validation with list"""
        with patch("src.examples.red.environment.PokemonRedEngine"):
            env = PokemonRedEnvironment()

            call = EnvToolCall(tool="press_button", args={"button": "A"})
            validated = env.validate_tool_calls([call])

            assert validated == call

    def test_validate_tool_calls_nested_list(self):
        """Test tool call validation with nested list"""
        with patch("src.examples.red.environment.PokemonRedEngine"):
            env = PokemonRedEnvironment()

            call = EnvToolCall(tool="press_button", args={"button": "A"})
            validated = env.validate_tool_calls([[call]])

            assert validated == call

    def test_validate_tool_calls_invalid_tool(self):
        """Test tool call validation with invalid tool"""
        with patch("src.examples.red.environment.PokemonRedEngine"):
            env = PokemonRedEnvironment()

            call = EnvToolCall(tool="invalid_tool", args={})
            with pytest.raises(ValueError, match="Unknown tool: invalid_tool"):
                env.validate_tool_calls(call)

    def test_validate_tool_calls_empty_list(self):
        """Test tool call validation with empty list"""
        with patch("src.examples.red.environment.PokemonRedEngine"):
            env = PokemonRedEnvironment()

            with pytest.raises(ValueError, match="empty list"):
                env.validate_tool_calls([])

    def test_validate_tool_calls_wrong_type(self):
        """Test tool call validation with wrong type"""
        with patch("src.examples.red.environment.PokemonRedEngine"):
            env = PokemonRedEnvironment()

            with pytest.raises(TypeError):
                env.validate_tool_calls("not_a_call")

    @patch("src.examples.red.environment.PokemonRedEngine")
    @pytest.mark.asyncio
    async def test_step_successful(self, mock_engine_class, mock_engine):
        """Test successful step execution"""
        mock_engine_class.return_value = mock_engine

        # Mock successful tool execution
        tool_result = ToolResult(
            ok=True,
            payload={
                "private": PokemonRedPrivateState(
                    reward_last_step=0.1,
                    total_reward=0.1,
                    terminated=False,
                    truncated=False,
                    step_count=1,
                ),
                "public": create_test_public_state(
                    map_id=3,
                    player_x=11,
                    player_y=8,
                    badges=0,
                    in_battle=False,
                    party_level=10,
                    party_hp_current=35,
                    party_hp_max=35,
                    party_xp=1000,
                    step_count=1,
                ),
            },
        )

        env = PokemonRedEnvironment()
        env._press_button_tool = AsyncMock(return_value=tool_result)

        call = EnvToolCall(tool="press_button", args={"button": "RIGHT"})
        obs = await env.step(call)

        assert obs["position"] == "Map03:(11,8)"
        assert obs["step_count"] == 1
        assert obs["total_reward"] == 0.1

    @patch("src.examples.red.environment.PokemonRedEngine")
    @pytest.mark.asyncio
    async def test_step_failed_tool(self, mock_engine_class, mock_engine):
        """Test step with failed tool execution"""
        mock_engine_class.return_value = mock_engine
        mock_engine._create_states.return_value = (
            PokemonRedPrivateState(
                reward_last_step=0.0,
                total_reward=0.0,
                terminated=False,
                truncated=False,
                step_count=0,
            ),
            create_test_public_state(
                map_id=3,
                player_x=10,
                player_y=8,
                badges=0,
                in_battle=False,
                party_level=10,
                party_hp_current=35,
                party_hp_max=35,
                party_xp=1000,
                step_count=0,
                error_info="Button press failed",
            ),
        )

        # Mock failed tool execution
        tool_result = ToolResult(ok=False, error="Invalid button", payload={"public": {}})

        env = PokemonRedEnvironment()
        env._press_button_tool = AsyncMock(return_value=tool_result)

        call = EnvToolCall(tool="press_button", args={"button": "INVALID"})
        obs = await env.step(call)

        # Should still return valid observation
        assert "position" in obs

    @patch("src.examples.red.environment.PokemonRedEngine")
    @pytest.mark.asyncio
    async def test_checkpoint(self, mock_engine_class, mock_engine):
        """Test environment checkpointing"""
        mock_engine_class.return_value = mock_engine
        mock_engine._serialize_engine.return_value = Mock(model_dump=lambda: {"test": "data"})
        mock_engine._create_states.return_value = (
            PokemonRedPrivateState(
                reward_last_step=0.0,
                total_reward=5.0,
                terminated=False,
                truncated=False,
                step_count=20,
            ),
            create_test_public_state(
                map_id=4,
                player_x=15,
                player_y=12,
                badges=1,
                in_battle=False,
                party_level=11,
                party_hp_current=40,
                party_hp_max=40,
                party_xp=1200,
                step_count=20,
            ),
        )

        env = PokemonRedEnvironment()
        obs = await env.checkpoint()

        assert "engine_snapshot_data" in obs
        assert obs["step_count"] == 20
        assert obs["total_reward"] == 5.0

    @pytest.mark.asyncio
    async def test_observation_callable(self):
        """Test observation callable functionality"""
        obs_callable = PokemonRedObservationCallable()

        priv_state = PokemonRedPrivateState(
            reward_last_step=0.1,
            total_reward=2.5,
            terminated=False,
            truncated=False,
            step_count=25,
        )

        pub_state = create_test_public_state(
            map_id=5,
            player_x=20,
            player_y=15,
            badges=3,  # 2 badges set
            in_battle=True,
            party_level=15,
            party_hp_current=25,
            party_hp_max=50,
            party_xp=5000,
            step_count=25,
            error_info="Test error",
        )

        obs = await obs_callable.get_observation(pub_state, priv_state)

        assert obs["position"] == "Map05:(20,15)"
        assert obs["badges_earned"] == 2  # bin(3).count('1')
        assert obs["badges_bitfield"] == 3
        assert obs["hp_status"] == "HP: 25/50 (50%)"
        assert obs["party_level"] == 15
        assert obs["in_battle"] is True
        assert obs["step_count"] == 25
        assert obs["total_reward"] == 2.5
        assert obs["error"] == "Test error"


class TestPressButtonTool:
    """Test the press button tool"""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine for tool testing"""
        engine = Mock()
        engine._step_engine = AsyncMock(
            return_value=(
                PokemonRedPrivateState(
                    reward_last_step=0.0,
                    total_reward=0.0,
                    terminated=False,
                    truncated=False,
                    step_count=1,
                ),
                create_test_public_state(
                    map_id=3,
                    player_x=10,
                    player_y=8,
                    badges=0,
                    in_battle=False,
                    party_level=10,
                    party_hp_current=35,
                    party_hp_max=35,
                    party_xp=1000,
                    step_count=1,
                ),
            )
        )
        return engine

    @pytest.mark.asyncio
    async def test_press_button_tool_success(self, mock_engine):
        """Test successful button press tool execution"""
        tool = PressButtonTool(mock_engine)

        call = EnvToolCall(tool="press_button", args={"button": "A", "frames": 2})
        result = await tool(call)

        assert result.ok is True
        assert "public" in result.payload
        assert "private" in result.payload
        mock_engine._step_engine.assert_called_once_with({"button": "A", "frames": 2})

    @pytest.mark.asyncio
    async def test_press_button_tool_invalid_args(self, mock_engine):
        """Test button press tool with invalid arguments"""
        tool = PressButtonTool(mock_engine)
        mock_engine._create_states.return_value = (Mock(), Mock())

        # Missing required button argument
        call = EnvToolCall(tool="press_button", args={"frames": 1})
        result = await tool(call)

        assert result.ok is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_press_button_tool_engine_error(self, mock_engine):
        """Test button press tool when engine raises error"""
        tool = PressButtonTool(mock_engine)
        mock_engine._step_engine.side_effect = Exception("Engine error")
        mock_engine._create_states.return_value = (Mock(), Mock())

        call = EnvToolCall(tool="press_button", args={"button": "A"})
        result = await tool(call)

        assert result.ok is False
        assert "Engine error" in result.error


# Helper function to create properly structured PokemonRedPublicState
def create_test_public_state(
    map_id: int = 3,
    player_x: int = 10,
    player_y: int = 8,
    badges: int = 0,
    in_battle: bool = False,
    party_level: int = 10,
    party_hp_current: int = 35,
    party_hp_max: int = 35,
    party_xp: int = 1000,
    step_count: int = 0,
    error_info: str = None,
) -> PokemonRedPublicState:
    """Create a properly structured PokemonRedPublicState for testing"""

    # Create structured components
    world = GameWorldState(map_id=map_id, player_x=player_x, player_y=player_y)

    progress = PlayerProgressState(
        badges=badges,
        badge_count=badges,  # badge_count should match badges
        money=3000,
        step_count=step_count,
    )

    system = GameSystemState(
        in_battle=in_battle,
        battle_outcome=0,
        menu_state=1,
        text_box_active=False,
        warp_flag=207,
    )

    # Create party if stats are provided
    party = []
    if party_level > 0:
        pokemon = PokemonData(
            species_id=25,  # Pikachu
            level=party_level,
            hp_current=party_hp_current,
            hp_max=party_hp_max,
            xp=party_xp,
            hp_percentage=party_hp_current / party_hp_max * 100.0 if party_hp_max > 0 else 0.0,
        )
        party.append(pokemon)

    return PokemonRedPublicState(
        world=world,
        progress=progress,
        party=party,
        inventory=[],
        system=system,
        error_info=error_info,
    )
