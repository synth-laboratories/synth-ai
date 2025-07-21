from .memory_map import *
from typing import Dict, Any, List


def get_byte(memory, addr: int) -> int:
    """Get single byte from memory"""
    try:
        return memory[addr]
    except (IndexError, TypeError):
        return 0


def get_word(memory, addr: int) -> int:
    """Get 16-bit word from memory (little endian)"""
    return get_byte(memory, addr) | (get_byte(memory, addr + 1) << 8)


def get_3byte_int(memory, addr: int) -> int:
    """Get 24-bit integer from memory for XP values"""
    return (
        get_byte(memory, addr)
        | (get_byte(memory, addr + 1) << 8)
        | (get_byte(memory, addr + 2) << 16)
    )


def get_bcd_3byte(memory, addr: int) -> int:
    """Get 3-byte BCD (Binary Coded Decimal) value for money"""
    byte1 = get_byte(memory, addr)
    byte2 = get_byte(memory, addr + 1)
    byte3 = get_byte(memory, addr + 2)
    return (
        (byte1 >> 4) * 100000
        + (byte1 & 0xF) * 10000
        + (byte2 >> 4) * 1000
        + (byte2 & 0xF) * 100
        + (byte3 >> 4) * 10
        + (byte3 & 0xF)
    )


def extract_party_pokemon(memory) -> List[Dict[str, Any]]:
    """Extract detailed information for each Pokemon in party"""
    party_count = get_byte(memory, PARTY_COUNT)
    party = []

    for i in range(party_count):
        # Basic Pokemon info
        species = get_byte(memory, PARTY_SPECIES + i)
        level = get_byte(memory, PARTY_LEVELS + i)

        # HP (2 bytes each Pokemon)
        hp_current = get_word(memory, PARTY_HP_CURRENT + (i * 2))
        hp_max = get_word(memory, PARTY_HP_MAX + (i * 2))

        # XP (3 bytes each Pokemon)
        xp = get_3byte_int(memory, PARTY_XP + (i * 3))

        pokemon = {
            "species_id": species,
            "level": level,
            "hp_current": hp_current,
            "hp_max": hp_max,
            "xp": xp,
            "hp_percentage": round((hp_current / hp_max * 100) if hp_max > 0 else 0, 1),
        }
        party.append(pokemon)

    return party


def extract_inventory(memory) -> List[Dict[str, Any]]:
    """Extract inventory items"""
    item_count = get_byte(memory, INVENTORY_COUNT)
    inventory = []

    for i in range(min(item_count, 20)):  # Max 20 items
        item_id = get_byte(memory, INVENTORY_START + (i * 2))
        quantity = get_byte(memory, INVENTORY_START + (i * 2) + 1)

        if item_id > 0:  # Valid item
            inventory.append({"item_id": item_id, "quantity": quantity})

    return inventory


def extract_game_state(memory) -> Dict[str, Any]:
    """Extract comprehensive game state from Game Boy memory"""
    # Get party and inventory details
    party = extract_party_pokemon(memory)
    inventory = extract_inventory(memory)

    # Get money
    money = get_bcd_3byte(memory, MONEY)

    # Basic game state
    state = {
        "map_id": get_byte(memory, MAP_ID),
        "player_x": get_byte(memory, PLAYER_X),
        "player_y": get_byte(memory, PLAYER_Y),
        "badges": get_byte(memory, BADGE_FLAGS),
        "in_battle": get_byte(memory, IN_BATTLE_FLAG) > 0,
        "battle_outcome": get_byte(memory, BATTLE_OUTCOME),
        "money": money,
        "menu_state": get_byte(memory, MENU_STATE),
        "warp_flag": get_byte(memory, WARP_FLAG),
        "text_box_active": get_byte(memory, TEXT_BOX_ACTIVE) > 0,
        # Detailed party and inventory
        "party_count": len(party),
        "party_pokemon": party,
        "inventory_count": len(inventory),
        "inventory_items": inventory,
        # Legacy fields for compatibility (use first Pokemon if available)
        "party_level": party[0]["level"] if party else 0,
        "party_hp_current": party[0]["hp_current"] if party else 0,
        "party_hp_max": party[0]["hp_max"] if party else 0,
        "party_xp": party[0]["xp"] if party else 0,
        "inventory_count_legacy": get_byte(memory, INVENTORY_COUNT),
    }

    return state


def get_badge_count(badges_byte: int) -> int:
    """Count number of badges from bitfield"""
    return bin(badges_byte).count("1")


def format_position(x: int, y: int, map_id: int) -> str:
    """Format position as readable string"""
    return f"Map{map_id:02X}:({x},{y})"


def format_hp_status(current: int, max_hp: int) -> str:
    """Format HP as readable string"""
    if max_hp == 0:
        return "HP: Unknown"
    percentage = (current / max_hp) * 100
    return f"HP: {current}/{max_hp} ({percentage:.0f}%)"
