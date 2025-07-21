from synth_ai.environments.examples.red.engine_helpers.state_extraction import (
    extract_game_state,
    get_badge_count,
    format_position,
    format_hp_status,
    get_byte,
    get_word,
    get_3byte_int,
)
from synth_ai.environments.examples.red.engine_helpers.memory_map import *


class TestMemoryExtraction:
    """Test memory extraction functions"""

    def test_get_byte(self):
        """Test single byte extraction"""
        memory = bytearray([0x00, 0x42, 0xFF, 0x80])
        assert get_byte(memory, 0) == 0x00
        assert get_byte(memory, 1) == 0x42
        assert get_byte(memory, 2) == 0xFF
        assert get_byte(memory, 3) == 0x80

        # Test bounds checking
        assert get_byte(memory, 100) == 0

    def test_get_word(self):
        """Test 16-bit word extraction (little endian)"""
        memory = bytearray([0x34, 0x12, 0xFF, 0x00])
        assert get_word(memory, 0) == 0x1234
        assert get_word(memory, 2) == 0x00FF

    def test_get_3byte_int(self):
        """Test 24-bit integer extraction for XP values"""
        memory = bytearray([0x56, 0x34, 0x12, 0x00])
        assert get_3byte_int(memory, 0) == 0x123456

    def test_extract_game_state(self):
        """Test full game state extraction"""
        # Create mock Game Boy memory
        memory = bytearray(0x10000)  # 64KB

        # Set test values at known addresses
        memory[MAP_ID] = 0x03  # Pewter City
        memory[PLAYER_X] = 10  # X position
        memory[PLAYER_Y] = 8  # Y position
        memory[BADGE_FLAGS] = 0x01  # Boulder Badge
        memory[IN_BATTLE_FLAG] = 0  # Not in battle
        memory[PARTY_COUNT] = 1  # One Pokemon in party
        memory[PARTY_LEVELS] = 12  # Level 12
        memory[PARTY_HP_CURRENT] = 35  # Current HP (low byte)
        memory[PARTY_HP_CURRENT + 1] = 0  # Current HP (high byte)
        memory[PARTY_HP_MAX] = 42  # Max HP (low byte)
        memory[PARTY_HP_MAX + 1] = 0  # Max HP (high byte)
        memory[PARTY_XP] = 0x40  # XP (low byte)
        memory[PARTY_XP + 1] = 0x42  # XP (mid byte)
        memory[PARTY_XP + 2] = 0x0F  # XP (high byte)

        state = extract_game_state(memory)

        assert state["map_id"] == 0x03
        assert state["player_x"] == 10
        assert state["player_y"] == 8
        assert state["badges"] == 0x01
        assert state["in_battle"] == False
        assert state["party_level"] == 12
        assert state["party_hp_current"] == 35
        assert state["party_hp_max"] == 42
        assert state["party_xp"] == 0x0F4240  # 1000000 in decimal

    def test_get_badge_count(self):
        """Test badge counting from bitfield"""
        assert get_badge_count(0x00) == 0  # No badges
        assert get_badge_count(0x01) == 1  # Boulder Badge
        assert get_badge_count(0x03) == 2  # Boulder + Cascade
        assert get_badge_count(0xFF) == 8  # All badges
        assert get_badge_count(0x55) == 4  # Every other badge

    def test_format_position(self):
        """Test position formatting"""
        assert format_position(10, 8, 3) == "Map03:(10,8)"
        assert format_position(0, 0, 255) == "MapFF:(0,0)"

    def test_format_hp_status(self):
        """Test HP status formatting"""
        assert format_hp_status(35, 50) == "HP: 35/50 (70%)"
        assert format_hp_status(0, 35) == "HP: 0/35 (0%)"
        assert format_hp_status(35, 35) == "HP: 35/35 (100%)"
        assert format_hp_status(10, 0) == "HP: Unknown"

    def test_memory_addresses_valid(self):
        """Test that all memory addresses are valid Game Boy addresses"""
        addresses = [
            BADGE_FLAGS,
            MAP_ID,
            PLAYER_X,
            PLAYER_Y,
            IN_BATTLE_FLAG,
            BATTLE_OUTCOME,
            PARTY_LEVELS,
            PARTY_HP_CURRENT,
            PARTY_HP_MAX,
            PARTY_XP,
            INVENTORY_COUNT,
            INVENTORY_START,
            MENU_STATE,
            WARP_FLAG,
        ]

        for addr in addresses:
            assert 0x8000 <= addr <= 0xFFFF, f"Address {hex(addr)} outside Game Boy RAM range"
