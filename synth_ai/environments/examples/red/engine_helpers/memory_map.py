# Pokemon Red memory addresses for state extraction
BADGE_FLAGS = 0xD356  # bit-field for badges
MAP_ID = 0xD35E  # current map ID
PLAYER_X = 0xD362  # player X coordinate
PLAYER_Y = 0xD361  # player Y coordinate
IN_BATTLE_FLAG = 0xD057  # battle state flag
BATTLE_OUTCOME = 0xD089  # 0=ongoing, 1=win, 2=lose

# Party Pokemon data (up to 6 Pokemon)
PARTY_COUNT = 0xD163  # number of Pokemon in party (0-6)
PARTY_SPECIES = 0xD164  # species of each Pokemon (6 bytes)
PARTY_HP_CURRENT = 0xD16C  # current HP of each Pokemon (2 bytes each, 12 bytes total)
PARTY_HP_MAX = 0xD188  # max HP of each Pokemon (2 bytes each, 12 bytes total)
PARTY_LEVELS = 0xD18C  # level of each Pokemon (6 bytes)
PARTY_XP = 0xD179  # XP of each Pokemon (3 bytes each, 18 bytes total)

# Player data
MONEY = 0xD347  # player money (3 bytes, BCD format)
PLAYER_NAME = 0xD158  # player name (up to 11 bytes)

# Inventory data
INVENTORY_COUNT = 0xD31D  # number of items in bag
INVENTORY_START = 0xD31E  # start of item/quantity pairs (up to 20 items, 2 bytes each)

# Game state flags
MENU_STATE = 0xCC26  # menu state flags
WARP_FLAG = 0xD36C  # warp/transition flags
TEXT_BOX_ACTIVE = 0xCD3D  # text box display flag
