from typing import Dict

import crafter.constants as C

# Map each action name to its corresponding index in the crafter package
CRAFTER_ACTION_MAP: Dict[str, int] = {action_name: idx for idx, action_name in enumerate(C.actions)}
