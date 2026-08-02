"""
WHY A* SPECIFICALLY:
A* is the standard, well-understood algorithm for "find the shortest path
from A to B while avoiding obstacles" -- it's what's used under the hood in
most game pathfinding and a lot of real robotics navigation stacks. It
guarantees the shortest possible path (given the grid resolution) rather
than just "a" path, and it's fast enough to run per-request without
needing to precompute anything.

HOW THIS WORKS AT A HIGH LEVEL:
1. Lay an imaginary grid over the area between the drone's start point and
   the landing point (plus some margin).
2. Mark grid cells as blocked if they're within `obstacle_radius` of any
   placed obstacle.
3. Search the grid for the shortest walkable path from start to goal,
   allowing 8-directional movement (not just up/down/left/right) for more
   natural-looking diagonal paths.
4. Convert the resulting grid path back into real-world (x, z) coordinates.

This operates in the same simplified 2D (x, z) "scene space" that
obstacles/landing/path objects already use (see app/models/scene_object.py)
-- altitude/3D pathfinding is intentionally out of scope for this first
version, noted as a possible future extension.
"""

import heapq
import math
from typing import List, Optional, Tuple

Point = Tuple[float, float]


def _world_to_grid(point: Point, origin: Point, cell_size: float) -> Tuple[int, int]:
    return (
        round((point[0] - origin[0]) / cell_size),
        round((point[1] - origin[1]) / cell_size),
    )


def _grid_to_world(cell: Tuple[int, int], origin: Point, cell_size: float) -> Point:
    return (origin[0] + cell[0] * cell_size, origin[1] + cell[1] * cell_size)


def _is_blocked(
    world_point: Point, obstacles: List[Point], obstacle_radius: float
) -> bool:
    for ox, oz in obstacles:
        if math.hypot(world_point[0] - ox, world_point[1] - oz) <= obstacle_radius:
            return True
    return False


# 8-directional movement: 4 straight neighbors (cost 1) + 4 diagonal
# neighbors (cost sqrt(2)) -- this is what makes the resulting path look
# like a natural diagonal line instead of a blocky staircase.
_NEIGHBORS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]


def plan_path(
    start: Point,
    goal: Point,
    obstacles: List[Point],
    cell_size: float = 0.5,
    obstacle_radius: float = 0.6,
    margin: float = 3.0,
) -> Optional[List[Point]]:
    """
    Returns a list of (x, z) waypoints from start to goal that avoids every
    obstacle by at least `obstacle_radius`, or None if no path exists
    (e.g. the goal is fully enclosed by obstacles).
    """
    if _is_blocked(start, obstacles, obstacle_radius):
        raise ValueError("Start point is inside an obstacle's safety radius")
    if _is_blocked(goal, obstacles, obstacle_radius):
        raise ValueError("Goal (landing point) is inside an obstacle's safety radius")

    # The grid only needs to span the area around start/goal/obstacles,
    # plus a margin -- no need to grid the whole infinite plane.
    all_x = [start[0], goal[0]] + [o[0] for o in obstacles]
    all_z = [start[1], goal[1]] + [o[1] for o in obstacles]
    origin = (min(all_x) - margin, min(all_z) - margin)
    max_point = (max(all_x) + margin, max(all_z) + margin)

    grid_width = int((max_point[0] - origin[0]) / cell_size) + 1
    grid_height = int((max_point[1] - origin[1]) / cell_size) + 1

    start_cell = _world_to_grid(start, origin, cell_size)
    goal_cell = _world_to_grid(goal, origin, cell_size)

    def in_bounds(cell):
        return 0 <= cell[0] < grid_width and 0 <= cell[1] < grid_height

    def blocked(cell):
        return _is_blocked(_grid_to_world(cell, origin, cell_size), obstacles, obstacle_radius)

    def heuristic(a, b):
        # Euclidean distance -- admissible (never overestimates) for an
        # 8-directional grid, which is what makes A* guarantee shortest path.
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # ---------- Standard A* search ----------
    open_heap = [(0.0, start_cell)]
    came_from = {}
    g_score = {start_cell: 0.0}
    visited = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal_cell:
            # Reconstruct the path by walking back through came_from.
            path_cells = [current]
            while current in came_from:
                current = came_from[current]
                path_cells.append(current)
            path_cells.reverse()
            return [_grid_to_world(c, origin, cell_size) for c in path_cells]

        for dx, dz, cost in _NEIGHBORS:
            neighbor = (current[0] + dx, current[1] + dz)
            if not in_bounds(neighbor) or neighbor in visited or blocked(neighbor):
                continue
            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal_cell)
                heapq.heappush(open_heap, (f_score, neighbor))

    return None  # goal genuinely unreachable given the obstacles


def path_distance_meters(path: List[Point]) -> float:
    """Total straight-segment length of a computed path, in scene units
    (treated as meters for estimation purposes -- see the endpoint's
    docstring for the same caveat about scene-unit-to-meter mapping)."""
    total = 0.0
    for i in range(1, len(path)):
        total += math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
    return total


# ============================================================================
# 3D GRID-BASED PATH PLANNING (extension of the 2D planner above)
#
# WHY A SEPARATE FUNCTION INSTEAD OF CHANGING plan_path():
# plan_path() above is already used by the live /plan-path endpoint and
# fully covered by test_path_planner.py. SceneObject rows already store a
# real y (height) coordinate (see app/models/scene_object.py) that the 2D
# planner has always ignored -- this adds altitude-aware planning as a
# NEW function/endpoint, so every existing caller of plan_path() keeps
# working completely unchanged; nothing about the 2D planner's behavior,
# signature, or tests is touched.
#
# HOW IT EXTENDS THE SAME A* APPROACH:
# Identical idea to the 2D version -- grid the search space, mark cells
# blocked within obstacle_radius of an obstacle, run A* -- just with a
# 3D grid and 26-directional movement (the 3D generalization of the 2D
# version's 8-directional movement: 6 face neighbors at cost 1, 12 edge
# neighbors at cost sqrt(2), 8 corner neighbors at cost sqrt(3)) and a 3D
# Euclidean heuristic, which stays admissible for the same reason the 2D
# Euclidean heuristic does (it never overestimates the true remaining
# distance on this grid).
# ============================================================================

Point3D = Tuple[float, float, float]


def _world_to_grid_3d(point: Point3D, origin: Point3D, cell_size: float) -> Tuple[int, int, int]:
    return (
        round((point[0] - origin[0]) / cell_size),
        round((point[1] - origin[1]) / cell_size),
        round((point[2] - origin[2]) / cell_size),
    )


def _grid_to_world_3d(cell: Tuple[int, int, int], origin: Point3D, cell_size: float) -> Point3D:
    return (
        origin[0] + cell[0] * cell_size,
        origin[1] + cell[1] * cell_size,
        origin[2] + cell[2] * cell_size,
    )


def _is_blocked_3d(world_point: Point3D, obstacles: List[Point3D], obstacle_radius: float) -> bool:
    for ox, oy, oz in obstacles:
        if math.dist(world_point, (ox, oy, oz)) <= obstacle_radius:
            return True
    return False


def _build_neighbors_3d():
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                cost = math.sqrt(dx * dx + dy * dy + dz * dz)
                neighbors.append((dx, dy, dz, cost))
    return neighbors


_NEIGHBORS_3D = _build_neighbors_3d()  # 26 directions: 6 face (cost 1) + 12 edge (cost sqrt2) + 8 corner (cost sqrt3)


def plan_path_3d(
    start: Point3D,
    goal: Point3D,
    obstacles: List[Point3D],
    cell_size: float = 0.5,
    obstacle_radius: float = 0.6,
    margin: float = 3.0,
) -> Optional[List[Point3D]]:
    """
    3D counterpart to plan_path() above -- returns a list of (x, y, z)
    waypoints from start to goal that avoids every obstacle by at least
    `obstacle_radius`, allowing altitude changes, or None if no path
    exists. Same grid-and-A* approach and same parameter meanings as the
    2D version, just with a third (y/altitude) dimension.
    """
    if _is_blocked_3d(start, obstacles, obstacle_radius):
        raise ValueError("Start point is inside an obstacle's safety radius")
    if _is_blocked_3d(goal, obstacles, obstacle_radius):
        raise ValueError("Goal (landing point) is inside an obstacle's safety radius")

    all_x = [start[0], goal[0]] + [o[0] for o in obstacles]
    all_y = [start[1], goal[1]] + [o[1] for o in obstacles]
    all_z = [start[2], goal[2]] + [o[2] for o in obstacles]
    origin = (min(all_x) - margin, min(all_y) - margin, min(all_z) - margin)
    max_point = (max(all_x) + margin, max(all_y) + margin, max(all_z) + margin)

    grid_size = (
        int((max_point[0] - origin[0]) / cell_size) + 1,
        int((max_point[1] - origin[1]) / cell_size) + 1,
        int((max_point[2] - origin[2]) / cell_size) + 1,
    )

    start_cell = _world_to_grid_3d(start, origin, cell_size)
    goal_cell = _world_to_grid_3d(goal, origin, cell_size)

    def in_bounds(cell):
        return (
            0 <= cell[0] < grid_size[0]
            and 0 <= cell[1] < grid_size[1]
            and 0 <= cell[2] < grid_size[2]
        )

    def blocked(cell):
        return _is_blocked_3d(_grid_to_world_3d(cell, origin, cell_size), obstacles, obstacle_radius)

    def heuristic(a, b):
        return math.dist(a, b)

    open_heap = [(0.0, start_cell)]
    came_from = {}
    g_score = {start_cell: 0.0}
    visited = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal_cell:
            path_cells = [current]
            while current in came_from:
                current = came_from[current]
                path_cells.append(current)
            path_cells.reverse()
            return [_grid_to_world_3d(c, origin, cell_size) for c in path_cells]

        for dx, dy, dz, cost in _NEIGHBORS_3D:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            if not in_bounds(neighbor) or neighbor in visited or blocked(neighbor):
                continue
            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal_cell)
                heapq.heappush(open_heap, (f_score, neighbor))

    return None  # goal genuinely unreachable given the obstacles


def path_distance_meters_3d(path: List[Point3D]) -> float:
    """3D counterpart to path_distance_meters() above."""
    total = 0.0
    for i in range(1, len(path)):
        total += math.dist(path[i], path[i - 1])
    return total
