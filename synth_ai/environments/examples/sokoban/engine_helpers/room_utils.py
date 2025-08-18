import copy
import marshal
from collections import deque
from typing import Tuple

import numpy as np

# --- ADDED CONSTANT ---
MAX_GENERATION_RETRIES = 100
# --- END ADDED CONSTANT ---

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


def get_shortest_action_path(room_fixed, room_state, MAX_DEPTH=100):
    """
    Get the shortest action path to push all boxes to the target spots.
    Use BFS to find the shortest path.
    NOTE currently only support one player, only one shortest solution
    =========================================================
    Parameters:
        room_state (np.ndarray): the state of the room
            - 0: wall
            - 1: empty space
            - 2: box target
            - 3: box on target
            - 4: box not on target
            - 5: player
        room_fixed (np.ndarray): the fixed part of the room
            - 0: wall
            - 1: empty space
            - 2: box target
        MAX_DEPTH (int): the maximum depth of the search
    =========================================================
    Returns:
        action_sequence (list): the action sequence to push all boxes to the target spots
    """

    # BFS queue stores (room_state, path)
    queue = deque([(copy.deepcopy(room_state), [])])
    explored_states = set()

    # Possible moves: up, down, left, right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    actions = [1, 2, 3, 4]  # Corresponding action numbers

    while queue:
        room_state, path = queue.popleft()
        if len(path) > MAX_DEPTH:
            return []  # No solution found

        # reduce the search space by checking if the state has been explored
        state_tohash = marshal.dumps(room_state)
        if state_tohash in explored_states:
            continue
        explored_states.add(state_tohash)

        # get information of the room
        player_pos = tuple(np.argwhere(room_state == 5)[0])
        boxes_on_target = set(map(tuple, np.argwhere((room_state == 3))))
        boxes_not_on_target = set(map(tuple, np.argwhere((room_state == 4))))
        boxes = boxes_on_target | boxes_not_on_target

        # Check if all boxes are on targets
        if not boxes_not_on_target:
            return path

        # Try each direction
        for move, action in zip(moves, actions):
            new_room_state = copy.deepcopy(room_state)
            new_player_pos = (player_pos[0] + move[0], player_pos[1] + move[1])

            # Check is new player position is wall or out of bound
            if (
                new_player_pos[0] < 0
                or new_player_pos[0] >= room_fixed.shape[0]
                or new_player_pos[1] < 0
                or new_player_pos[1] >= room_fixed.shape[1]
                or room_fixed[new_player_pos] == 0
            ):
                continue

            # If there's a box, check if we can push it
            if new_player_pos in boxes:
                box_pos = new_player_pos  # the original box position
                new_box_pos = (new_player_pos[0] + move[0], new_player_pos[1] + move[1])

                # Can't push if hitting wall or another box or out of bound
                if (
                    room_fixed[new_box_pos] == 0
                    or new_box_pos in boxes
                    or new_box_pos[0] < 0
                    or new_box_pos[0] >= room_fixed.shape[0]
                    or new_box_pos[1] < 0
                    or new_box_pos[1] >= room_fixed.shape[1]
                ):
                    continue

                # move the box

                new_room_state[box_pos] = room_fixed[box_pos]
                if room_fixed[new_box_pos] == 2:
                    new_room_state[new_box_pos] = 3
                else:
                    new_room_state[new_box_pos] = 4

            # player moves
            new_room_state[player_pos] = room_fixed[player_pos]
            new_room_state[new_player_pos] = 5
            queue.append((new_room_state, path + [action]))

    return []  # No solution found


# def plot_animation(imgs):
#     fig, ax = plt.subplots()
#     im = ax.imshow(imgs[0])
#     def init():
#         im.set_data(imgs[0])
#         return [im]
#     def update(i):
#         im.set_data(imgs[i])
#         return [im]
#     ani = animation.FuncAnimation(fig, update, frames=len(imgs), init_func=init, blit=True)
#     return ani


# def plot_animation(imgs):
#     height, width = imgs[0].shape[:2]
#     fig = plt.figure(figsize=(width / 100, height / 100), dpi=500)

#     ax = fig.add_axes([0, 0, 1, 1])

#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_frame_on(False)

#     im = ax.imshow(imgs[0])

#     def init():
#         im.set_data(imgs[0])
#         return [im]

#     def update(i):
#         im.set_data(imgs[i])
#         return [im]

#     ani = animation.FuncAnimation(
#         fig, update, frames=len(imgs), init_func=init, blit=True
#     )
#     return ani


# def solve_sokoban(env, saved_animation_path):
#     """
#     Solve the given sokoban environment and save the animation
#     """
#     actions = get_shortest_action_path(env.room_fixed, env.room_state)
#     print(f"Found {len(actions)} actions: {actions}")
#     imgs = []
#     img_before_action = env.render("rgb_array")
#     imgs.append(img_before_action)
#     for action in actions:
#         env.step(action)
#         img_after_action = env.render("rgb_array")
#         imgs.append(img_after_action)
#     ani = plot_animation(imgs)
#     ani.save(saved_animation_path)


def add_random_player_movement(
    room_state: np.ndarray,
    room_structure: np.ndarray,
    rng: np.random.Generator,
    move_probability=0.5,
    continue_probability=0.5,
    max_steps=3,
) -> np.ndarray:
    """
    Randomly move the player after reverse_playing.
    Uses the provided RNG instance.
    """
    # Check if we should move the player at all
    if rng.random() > move_probability:
        return room_state

    # Find player position
    player_pos_coords = np.where(room_state == 5)
    if len(player_pos_coords[0]) == 0:
        # Should not happen if called correctly, but handle gracefully
        print("Warning: Player not found in room_state during add_random_player_movement.")
        return room_state
    player_pos = np.array([player_pos_coords[0][0], player_pos_coords[1][0]])

    # Keep track of previous positions to avoid moving back
    previous_positions = {tuple(player_pos)}

    # Make 1-3 random moves
    steps_taken = 0
    while steps_taken < max_steps:
        # Get all valid moves (can't move into walls or boxes)
        valid_moves = []
        for action in range(4):  # 0: up, 1: down, 2: left, 3: right
            change = CHANGE_COORDINATES[action]
            next_pos = player_pos + change

            # Check bounds first
            if not (
                0 <= next_pos[0] < room_state.shape[0] and 0 <= next_pos[1] < room_state.shape[1]
            ):
                continue

            # Check if next position is valid (empty space or target) and not a previous position
            if (
                room_state[next_pos[0], next_pos[1]] in [1, 2]
                and tuple(next_pos) not in previous_positions
            ):
                valid_moves.append((action, next_pos))

        # If no valid moves, break
        if not valid_moves:
            break

        # Choose a random valid move using the passed RNG
        move_index = rng.integers(0, len(valid_moves))
        chosen_action, next_pos = valid_moves[move_index]

        # Move player
        room_state[player_pos[0], player_pos[1]] = room_structure[player_pos[0], player_pos[1]]
        room_state[next_pos[0], next_pos[1]] = 5

        # Update player position and track previous position
        player_pos = next_pos
        previous_positions.add(tuple(player_pos))

        steps_taken += 1

        # Decide whether to continue moving using passed RNG
        if steps_taken >= max_steps or rng.random() > continue_probability:
            break

    return room_state


"""
Following code is adapted from the nicely written gym_sokoban repo
"""


def generate_room(
    dim: Tuple[int, int] = (13, 13),
    initial_seed: int = 42,
    p_change_directions=0.35,
    num_steps=25,
    num_boxes=3,
    second_player=False,
    search_depth=100,
):
    """
    Generates a Sokoban room deterministically based on initial_seed.
    Retries generation with derived seeds if the first attempt fails (score=0).
    Uses numpy.random.default_rng for internal randomness.
    """

    for retry_offset in range(MAX_GENERATION_RETRIES):
        current_seed = initial_seed + retry_offset
        # Create a new RNG instance for this attempt, seeded deterministically
        rng = np.random.default_rng(current_seed)

        # Generate topology using the temporary RNG
        room = room_topology_generation(dim, p_change_directions, num_steps, rng)

        # Place boxes and player using the temporary RNG
        try:
            room = place_boxes_and_player(room, num_boxes, second_player, rng)
        except RuntimeError as e:
            # This can happen if not enough space; treat as generation failure for this seed
            print(f"Warning: place_boxes_and_player failed for seed {current_seed}: {e}")
            continue  # Try next seed offset

        # Prepare for reverse playing
        room_structure = np.copy(room)
        room_structure[room_structure == 5] = 1
        room_state = room.copy()
        room_state[room_state == 2] = 4  # Mark initial boxes as 'not on target'

        # Perform reverse playing (doesn't need RNG currently)
        room_state_rev, box_mapping, action_sequence = reverse_playing(
            room_state, room_structure, search_depth
        )

        # If reverse playing failed (returned None), try next seed offset
        if room_state_rev is None:
            print(f"Warning: reverse_playing failed for seed {current_seed}.")
            continue

        # Check score
        score = box_displacement_score(box_mapping)
        if score > 0:
            # Successfully generated a potentially solvable room
            final_room_state = room_state_rev.copy()
            final_room_state[final_room_state == 3] = (
                4  # Ensure boxes are marked correctly after reverse play
            )

            # Add random player movement using the temporary RNG
            move_prob = 0.8 if score == 1 else 0.5
            final_room_state = add_random_player_movement(
                final_room_state,
                room_structure,
                rng,  # Pass the temporary RNG
                move_probability=move_prob,
                continue_probability=0.5,
                max_steps=3,
            )
            # print(
            #     f"Successfully generated room with score {score} using seed {current_seed} (offset {retry_offset})."
            # )
            return room_structure, final_room_state, box_mapping, action_sequence
        else:
            # Score is 0, log and continue to next retry offset

            pass
            # print(
            #     f"Warning: Generated room with score 0 for seed {current_seed} (offset {retry_offset}). Retrying..."
            # )

    # If loop completes without success
    raise RuntimeWarning(
        f"Generated Model with score == 0 after {MAX_GENERATION_RETRIES} attempts. "
        f"Parameters: initial_seed={initial_seed}, dim={dim}, "
        f"p_change_directions={p_change_directions}, num_steps={num_steps}, "
        f"num_boxes={num_boxes}, second_player={second_player}, "
        f"search_depth={search_depth}."
    )


def room_topology_generation(
    dim: Tuple[int, int],
    p_change_directions: float,
    num_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate room topology using the provided RNG instance."""
    dim_x, dim_y = dim

    # The ones in the mask represent all fields which will be set to floors
    # during the random walk. The centered one will be placed over the current
    # position of the walk.
    masks = [
        [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [1, 1, 0]],
        [[0, 0, 0], [0, 1, 1], [0, 1, 0]],
    ]

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    # Use rng.choice for single sample
    direction_index = rng.choice(len(directions))
    direction = directions[direction_index]

    # Starting position uses rng.integers
    position = np.array([rng.integers(1, dim_x - 1), rng.integers(1, dim_y - 1)])

    level = np.zeros(dim, dtype=int)

    for _ in range(num_steps):
        # Change direction randomly using rng.random
        if rng.random() < p_change_directions:
            direction_index = rng.choice(len(directions))
            direction = directions[direction_index]

        # Update position
        position = position + direction
        position[0] = max(min(position[0], dim_x - 2), 1)
        position[1] = max(min(position[1], dim_y - 2), 1)

        # Apply mask using rng.choice
        mask_index = rng.choice(len(masks))
        mask = masks[mask_index]
        mask_start = position - 1
        level[mask_start[0] : mask_start[0] + 3, mask_start[1] : mask_start[1] + 3] += mask

    level[level > 0] = 1
    level[:, [0, dim_y - 1]] = 0
    level[[0, dim_x - 1], :] = 0

    return level


def place_boxes_and_player(
    room: np.ndarray, num_boxes: int, second_player: bool, rng: np.random.Generator
) -> np.ndarray:
    """
    Places player and boxes using the provided RNG instance.
    Raises RuntimeError if not enough space.
    """
    # Get all available positions
    possible_indices = np.where(room == 1)
    possible_positions = list(zip(possible_indices[0], possible_indices[1]))
    num_possible_positions = len(possible_positions)
    num_players = 2 if second_player else 1

    if num_possible_positions <= num_boxes + num_players:
        raise RuntimeError(
            f"Not enough free spots ({num_possible_positions}) to place {num_players} player(s) and {num_boxes} boxes."
        )

    # Choose positions using rng.choice without replacement
    chosen_indices = rng.choice(num_possible_positions, size=num_boxes + num_players, replace=False)
    chosen_positions = [possible_positions[i] for i in chosen_indices]

    # Place player(s)
    player_positions = chosen_positions[:num_players]
    for pos in player_positions:
        room[pos] = 5

    # Place boxes
    box_positions = chosen_positions[num_players:]
    for pos in box_positions:
        room[pos] = 2  # Place as box target initially

    return room


# Global variables used for reverse playing.
explored_states = set()
num_boxes = 0
best_room_score = -1
best_room = None
best_box_mapping = None


def reverse_playing(room_state, room_structure, search_depth=100):
    """
    This function plays Sokoban reverse in a way, such that the player can
    move and pull boxes.
    It ensures a solvable level with all boxes not being placed on a box target.
    :param room_state:
    :param room_structure:
    :param search_depth:
    :return: 2d array, box mapping, action sequence
    """
    global \
        explored_states, \
        num_boxes, \
        best_room_score, \
        best_room, \
        best_box_mapping, \
        best_action_sequence

    # Box_Mapping is used to calculate the box displacement for every box
    box_mapping = {}
    box_locations = np.where(room_structure == 2)
    num_boxes = len(box_locations[0])
    for l in range(num_boxes):
        box = (box_locations[0][l], box_locations[1][l])
        box_mapping[box] = box

    # explored_states globally stores the best room state and score found during search
    explored_states = set()
    best_room_score = -1
    best_room = None
    best_box_mapping = box_mapping
    best_action_sequence = []

    depth_first_search(
        room_state,
        room_structure,
        box_mapping,
        box_swaps=0,
        last_pull=(-1, -1),
        ttl=search_depth,
        action_sequence=[],
    )

    return best_room, best_box_mapping, best_action_sequence


def depth_first_search(
    room_state,
    room_structure,
    box_mapping,
    box_swaps=0,
    last_pull=(-1, -1),
    ttl=300,
    action_sequence=[],
):
    """
    Searches through all possible states of the room.
    This is a recursive function, which stops if the ttl is reduced to 0 or
    over 1.000.000 states have been explored.
    :param room_state:
    :param room_structure:
    :param box_mapping:
    :param box_swaps:
    :param last_pull:
    :param ttl:
    :param action_sequence:
    :return:
    """
    global \
        explored_states, \
        num_boxes, \
        best_room_score, \
        best_room, \
        best_box_mapping, \
        best_action_sequence

    ttl -= 1
    if ttl <= 0 or len(explored_states) >= 300000:
        return

    state_tohash = marshal.dumps(room_state)

    # Only search this state, if it not yet has been explored
    if state_tohash not in explored_states:
        # Add current state and its score to explored states
        room_score = box_swaps * box_displacement_score(box_mapping)
        if np.where(room_state == 2)[0].shape[0] != num_boxes:
            room_score = 0

        if room_score > best_room_score:
            best_room = room_state.copy()
            best_room_score = room_score
            best_box_mapping = box_mapping.copy()
            best_action_sequence = action_sequence.copy()

        explored_states.add(state_tohash)

        for action in ACTION_LOOKUP.keys():
            # The state and box mapping need to be copied to ensure
            # every action starts from a similar state.

            # TODO: A tentitive try here to make less moves
            if action >= 4:
                continue

            room_state_next = room_state.copy()
            box_mapping_next = box_mapping.copy()

            room_state_next, box_mapping_next, last_pull_next = reverse_move(
                room_state_next, room_structure, box_mapping_next, last_pull, action
            )

            box_swaps_next = box_swaps
            if last_pull_next != last_pull:
                box_swaps_next += 1

            action_sequence_next = action_sequence + [action]
            # action_sequence_next = action_sequence + [(action, box_mapping_next != box_mapping)] # add whether a box is moved
            depth_first_search(
                room_state_next,
                room_structure,
                box_mapping_next,
                box_swaps_next,
                last_pull_next,
                ttl,
                action_sequence_next,
            )


def reverse_move(room_state, room_structure, box_mapping, last_pull, action):
    """
    Perform reverse action. Where all actions in the range [0, 3] correspond to
    push actions and the ones greater 3 are simmple move actions.
    :param room_state:
    :param room_structure:
    :param box_mapping:
    :param last_pull:
    :param action:
    :return:
    """
    player_position = np.where(room_state == 5)
    player_position = np.array([player_position[0][0], player_position[1][0]])

    change = CHANGE_COORDINATES[action % 4]
    next_position = player_position + change

    # Check if next position is an empty floor or an empty box target
    if room_state[next_position[0], next_position[1]] in [1, 2]:
        # Move player, independent of pull or move action.
        room_state[player_position[0], player_position[1]] = room_structure[
            player_position[0], player_position[1]
        ]
        room_state[next_position[0], next_position[1]] = 5

        # In addition try to pull a box if the action is a pull action
        if action < 4:
            possible_box_location = change[0] * -1, change[1] * -1
            possible_box_location += player_position

            if room_state[possible_box_location[0], possible_box_location[1]] in [3, 4]:
                # Perform pull of the adjacent box
                room_state[player_position[0], player_position[1]] = 3
                room_state[possible_box_location[0], possible_box_location[1]] = room_structure[
                    possible_box_location[0], possible_box_location[1]
                ]

                # Update the box mapping
                for k in box_mapping.keys():
                    if box_mapping[k] == (
                        possible_box_location[0],
                        possible_box_location[1],
                    ):
                        box_mapping[k] = (player_position[0], player_position[1])
                        last_pull = k

    return room_state, box_mapping, last_pull


def box_displacement_score(box_mapping):
    """
    Calculates the sum of all Manhattan distances, between the boxes
    and their origin box targets.
    :param box_mapping:
    :return:
    """
    score = 0

    for box_target in box_mapping.keys():
        box_location = np.array(box_mapping[box_target])
        box_target = np.array(box_target)
        dist = np.sum(np.abs(box_location - box_target))
        score += dist

    return score


TYPE_LOOKUP = {
    0: "wall",
    1: "empty space",
    2: "box target",
    3: "box on target",
    4: "box not on target",
    5: "player",
}

ACTION_LOOKUP = {
    0: "push up",
    1: "push down",
    2: "push left",
    3: "push right",
    4: "move up",
    5: "move down",
    6: "move left",
    7: "move right",
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
