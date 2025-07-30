import functools

import numpy as np
import opensimplex

from . import constants
from . import objects
from .config import WorldGenConfig


def generate_world(world, player):
    simplex = opensimplex.OpenSimplex(seed=world.random.randint(0, 2**31 - 1))
    tunnels = np.zeros(world.area, bool)
    for x in range(world.area[0]):
        for y in range(world.area[1]):
            _set_material(world, (x, y), player, tunnels, simplex)
    for x in range(world.area[0]):
        for y in range(world.area[1]):
            _set_object(world, (x, y), player, tunnels)


def _set_material(world, pos, player, tunnels, simplex):
    x, y = pos
    config = getattr(world, "_config", WorldGenConfig())
    simplex = functools.partial(_simplex, simplex)
    uniform = world.random.uniform

    # Clear area around spawn
    dist_from_spawn = np.sqrt((x - player.pos[0]) ** 2 + (y - player.pos[1]) ** 2)
    start = config.spawn_clear_radius - dist_from_spawn
    start += 2 * simplex(x, y, 8, 3)
    start = 1 / (1 + np.exp(-start))

    # Terrain generation
    water = simplex(x, y, 3, {15: 1, 5: 0.15}, False) + 0.1
    water -= 2 * start
    mountain = simplex(x, y, 0, {15: 1, 5: 0.3})
    mountain -= 4 * start + 0.3 * water

    if start > 0.5:
        world[x, y] = "grass"
    elif mountain > config.mountain_threshold:
        if (simplex(x, y, 6, 7) > 0.15 and mountain > 0.3) or simplex(
            x, y, 6, 5
        ) > config.cave_threshold:  # cave
            world[x, y] = "path"
        elif simplex(2 * x, y / 5, 7, 3) > 0.4:  # horizonal tunnel
            world[x, y] = "path"
            tunnels[x, y] = True
        elif simplex(x / 5, 2 * y, 7, 3) > 0.4:  # vertical tunnel
            world[x, y] = "path"
            tunnels[x, y] = True
        elif simplex(x, y, 1, 8) > config.coal_threshold and uniform() > (
            1 - config.coal_probability
        ):
            world[x, y] = "coal"
        elif simplex(x, y, 2, 6) > config.iron_threshold and uniform() > (
            1 - config.iron_probability
        ):
            world[x, y] = "iron"
        elif mountain > config.diamond_threshold and uniform() > (1 - config.diamond_probability):
            world[x, y] = "diamond"
        elif mountain > 0.3 and simplex(x, y, 6, 5) > config.lava_threshold:
            world[x, y] = "lava"
        else:
            world[x, y] = "stone"
    elif (config.sand_threshold - 0.1) < water <= config.sand_threshold and simplex(
        x, y, 4, 9
    ) > -0.2:
        world[x, y] = "sand"
    elif config.water_threshold < water:
        world[x, y] = "water"
    else:  # grassland
        if simplex(x, y, 5, 7) > config.tree_threshold and uniform() > (
            1 - config.tree_probability
        ):
            world[x, y] = "tree"
        else:
            world[x, y] = "grass"


def _set_object(world, pos, player, tunnels):
    x, y = pos
    config = getattr(world, "_config", WorldGenConfig())
    uniform = world.random.uniform
    dist = np.sqrt((x - player.pos[0]) ** 2 + (y - player.pos[1]) ** 2)
    material, _ = world[x, y]

    if material not in constants.walkable:
        pass
    elif (
        dist > config.cow_min_distance
        and material == "grass"
        and uniform() < config.cow_spawn_probability
    ):
        world.add(objects.Cow(world, (x, y)))
    elif (
        dist > config.zombie_min_distance
        and material == "grass"
        and uniform() < config.zombie_spawn_probability
    ):
        world.add(objects.Zombie(world, (x, y), player))
    elif material == "path" and tunnels[x, y] and uniform() < config.skeleton_spawn_probability:
        world.add(objects.Skeleton(world, (x, y), player))


def _simplex(simplex, x, y, z, sizes, normalize=True):
    if not isinstance(sizes, dict):
        sizes = {sizes: 1}
    value = 0
    for size, weight in sizes.items():
        if hasattr(simplex, "noise3d"):
            noise = simplex.noise3d(x / size, y / size, z)
        else:
            noise = simplex.noise3(x / size, y / size, z)
        value += weight * noise
    if normalize:
        value /= sum(sizes.values())
    return value
