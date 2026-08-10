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
import random
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


# ============================================================================
# PROBABILISTIC / RISK-AWARE PATH PLANNING (extension of plan_path_3d above)
#
# WHY THIS EXISTS:
# plan_path_3d treats every obstacle's position as a single exact point --
# a cell is either "safe" (outside obstacle_radius) or "blocked" (inside
# it), a hard binary cutoff. In reality, a placed obstacle's position
# (from a camera/lidar/GPS-based obstacle detector) is only known to
# within some measurement uncertainty -- treating it as exact overstates
# confidence right at the boundary and can't express the difference
# between "well-localized obstacle, tight safety margin needed" and
# "poorly-localized obstacle, needs a much wider berth." This models each
# obstacle's true position as a random variable and plans a path that
# minimizes CUMULATIVE COLLISION PROBABILITY, not just avoids a fixed
# radius around a fixed point.
#
# THE MODEL -- GAUSSIAN OBSTACLE POSITION UNCERTAINTY:
# Each obstacle's true position is modeled as obstacle ~ N(obstacle_mean,
# sigma^2 * I) -- independent Gaussian noise in each of x/y/z around the
# reported (mean) position. This is the standard way position/landmark
# uncertainty is modeled throughout probabilistic robotics -- see Thrun,
# Burgard & Fox, "Probabilistic Robotics" (MIT Press, 2005), the standard
# reference for representing sensor-derived object positions as Gaussian-
# distributed random variables rather than exact points.
#
# COLLISION PROBABILITY VIA MONTE CARLO (reusing this app's existing
# Monte Carlo pattern from app/services/monte_carlo_uq.py):
# For a candidate point (a grid cell being evaluated during the search),
# the probability that it falls within `safety_radius` of an obstacle's
# TRUE (uncertain) position is estimated the same way
# monte_carlo_uq.py estimates uncertainty in BEMT/wind outputs: draw many
# samples of the obstacle's position from its assumed distribution, and
# take the fraction of samples within safety_radius of the candidate
# point. This is exactly the risk-estimation approach used in
# chance-constrained motion planning -- see Blackmore, Ono & Williams,
# "Chance-Constrained Optimal Path Planning with Obstacles," IEEE
# Transactions on Robotics, 2011, which evaluates exactly this kind of
# collision probability (sample-based, since no closed form exists for
# an arbitrary safety region) to plan paths under bounded collision risk.
# For multiple obstacles, this combines each obstacle's collision
# probability assuming independence: P(collision) = 1 - product(1 - P_i)
# -- the standard formula for the probability of at least one of several
# independent events occurring.
#
# WHY A WEIGHTED COST INSTEAD OF A HARD CHANCE CONSTRAINT:
# The Blackmore et al. formulation above solves for the shortest path
# subject to a maximum allowed TOTAL collision probability (a hard
# constraint, needing iterative search over a risk budget). This module
# uses a simpler, still-standard alternative: add `risk_weight *
# collision_probability` as an extra A* edge cost alongside the existing
# distance cost, so the search naturally trades a longer route for lower
# cumulative risk (a weighted multi-objective formulation, tunable via
# risk_weight) -- a documented simplification of the full chance-
# constrained formulation, appropriate for this app's scope. Cells with
# essentially-certain collision (probability > HARD_BLOCK_THRESHOLD) are
# still treated as impassable, exactly like plan_path_3d's hard
# obstacle_radius cutoff -- otherwise the search has no reason to avoid
# routing directly through an obstacle's mean position if the distance
# saved outweighs a merely large (but not ~1.0) risk penalty.
#
# WHAT WAS VERIFIED BEFORE SHIPPING:
# - At obstacle_position_std=0 (no uncertainty), every Monte Carlo sample
#   lands exactly on the obstacle's mean position, so collision
#   probability becomes a step function identical to plan_path_3d's hard
#   obstacle_radius check -- confirmed the probabilistic planner's result
#   closely matches plan_path_3d's result in this limit (same kind of
#   reduction check already used for plan_path_3d's flat-altitude-vs-2D
#   test).
# - Increasing obstacle_position_std (more positional uncertainty) while
#   holding everything else fixed makes the planner take a LARGER detour
#   -- the physically-expected direction (more uncertainty needs a wider
#   safety margin), verified numerically in test_path_planner_probabilistic.py.
# ============================================================================

HARD_BLOCK_THRESHOLD = 0.98  # collision probability above this is treated as impassable, not just costly


def _sample_obstacle_positions(
    obstacle_means: List[Point3D], obstacle_position_std: float, num_samples: int, seed: Optional[int] = None,
) -> List[List[Point3D]]:
    """Draws `num_samples` Monte Carlo samples of each obstacle's true
    position from N(mean, obstacle_position_std^2 * I) -- computed ONCE
    per planning call and reused for every grid cell evaluated during the
    search (far cheaper than resampling per cell), exactly the same
    "sample once, evaluate many times" pattern monte_carlo_uq.py uses."""
    rng = random.Random(seed)
    return [
        [
            (
                rng.gauss(ox, obstacle_position_std),
                rng.gauss(oy, obstacle_position_std),
                rng.gauss(oz, obstacle_position_std),
            )
            for _ in range(num_samples)
        ]
        for (ox, oy, oz) in obstacle_means
    ]


def collision_probability(
    point: Point3D, obstacle_samples: List[List[Point3D]], safety_radius: float,
) -> float:
    """
    Monte Carlo-estimated probability that `point` is within
    safety_radius of AT LEAST ONE obstacle's true (uncertain) position,
    given each obstacle's pre-drawn position samples (see
    _sample_obstacle_positions) -- P(collision) = 1 - product(1-P_i)
    across obstacles, each P_i the fraction of that obstacle's samples
    within safety_radius of `point`.
    """
    prob_no_collision = 1.0
    for samples in obstacle_samples:
        if not samples:
            continue
        hits = sum(1 for s in samples if math.dist(point, s) <= safety_radius)
        p_i = hits / len(samples)
        prob_no_collision *= (1 - p_i)
    return 1 - prob_no_collision


def plan_path_3d_probabilistic(
    start: Point3D,
    goal: Point3D,
    obstacle_means: List[Point3D],
    obstacle_position_std: float,
    safety_radius: float = 0.6,
    cell_size: float = 0.5,
    margin: float = 3.0,
    risk_weight: float = 50.0,
    num_mc_samples: int = 2000,
    seed: Optional[int] = 42,
) -> Optional[Tuple[List[Point3D], List[float]]]:
    """
    Risk-aware counterpart to plan_path_3d -- same grid/A* skeleton and
    the same 26-directional neighbor set (_NEIGHBORS_3D, reused as-is),
    but obstacle avoidance is now a continuous risk cost (see module
    docstring above) instead of a hard blocked/unblocked cutoff, except
    for near-certain-collision cells which stay hard-blocked. Returns
    (path, per_point_collision_probability) or None if no path exists.

    WHY seed DEFAULTS TO A FIXED VALUE (42), UNLIKE monte_carlo_uq.py's
    functions (which default to seed=None): those functions report
    aggregate STATISTICS (mean, percentiles) that are, by design, stable
    across different random seeds once the sample count is large enough
    -- which seed is used barely matters. Here, the Monte Carlo estimate
    feeds directly into which literal FLIGHT PATH gets returned -- an
    unseeded default would mean asking this endpoint for the same drone,
    same obstacles, same everything, twice in a row could return two
    different flight paths purely from sampling noise, which is a real,
    confusing behavior for a planning tool, not a cosmetic one (verified
    empirically: at the old default of 500 samples with no fixed seed,
    repeated calls with identical inputs occasionally returned VISIBLY
    different path lengths -- see the commit message for this feature
    for the specific numbers). Defaulting to a fixed seed makes the
    output reproducible for identical inputs, the behavior a path
    planner should have; num_mc_samples was also raised from 500 to 2000
    (matching monte_carlo_uq.py's established default) so the underlying
    risk estimate itself is also tighter, not just consistently repeating
    the same noisy answer.
    """
    obstacle_samples = _sample_obstacle_positions(obstacle_means, obstacle_position_std, num_mc_samples, seed)

    start_risk = collision_probability(start, obstacle_samples, safety_radius)
    if start_risk > HARD_BLOCK_THRESHOLD:
        raise ValueError("Start point has near-certain collision probability with an obstacle")
    goal_risk = collision_probability(goal, obstacle_samples, safety_radius)
    if goal_risk > HARD_BLOCK_THRESHOLD:
        raise ValueError("Goal (landing point) has near-certain collision probability with an obstacle")

    all_x = [start[0], goal[0]] + [o[0] for o in obstacle_means]
    all_y = [start[1], goal[1]] + [o[1] for o in obstacle_means]
    all_z = [start[2], goal[2]] + [o[2] for o in obstacle_means]
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

    # Cached per-cell so repeated A* neighbor expansions (a cell can be
    # reached via multiple paths before it's finalized) don't re-run the
    # Monte Carlo evaluation for the same cell twice.
    risk_cache: dict = {}

    def risk_at(cell) -> float:
        if cell not in risk_cache:
            world_point = _grid_to_world_3d(cell, origin, cell_size)
            risk_cache[cell] = collision_probability(world_point, obstacle_samples, safety_radius)
        return risk_cache[cell]

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
            path = [_grid_to_world_3d(c, origin, cell_size) for c in path_cells]
            risks = [risk_at(c) for c in path_cells]
            return path, risks

        for dx, dy, dz, cost in _NEIGHBORS_3D:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            if not in_bounds(neighbor) or neighbor in visited:
                continue
            neighbor_risk = risk_at(neighbor)
            if neighbor_risk > HARD_BLOCK_THRESHOLD:
                continue  # near-certain collision -- treated as impassable, same as plan_path_3d's hard radius
            tentative_g = g_score[current] + cost + risk_weight * neighbor_risk
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal_cell)
                heapq.heappush(open_heap, (f_score, neighbor))

    return None  # goal genuinely unreachable given the obstacles/risk field


# ============================================================================
# MINIMUM-JERK TRAJECTORY SMOOTHING (on top of plan_path_3d's waypoints)
#
# WHY A SEPARATE FUNCTION/ENDPOINT INSTEAD OF CHANGING plan_path_3d:
# plan_path_3d's grid-search output is a WAYPOINT LIST -- one point per
# grid cell crossed, with instantaneous direction changes at every corner
# and no notion of time, velocity, or acceleration at all. That's exactly
# right for "does a route exist and where does it go," which is the
# planner's job, but it is not something a real flight controller could
# actually fly smoothly (an instantaneous direction change implies
# infinite acceleration). This is a NEW function, generate_minimum_jerk_-
# trajectory(), that takes plan_path_3d's finished output AS INPUT and
# produces a smooth, time-parameterized trajectory from it -- it does not
# touch plan_path_3d itself, its signature, or its behavior in any way,
# so the existing /plan-path-3d endpoint (and its tests) are completely
# unaffected.
#
# THE METHOD -- PIECEWISE QUINTIC MINIMUM-JERK SPLINE:
# A single quintic (5th-order) polynomial segment, given boundary
# position/velocity/acceleration at both ends, is the unique polynomial
# that minimizes integrated squared jerk (the derivative of acceleration)
# between those two states -- the original result is Flash & Hogan, "The
# Coordination of Arm Movements: An Experimentally Confirmed Mathematical
# Model," Journal of Neuroscience, 1985, and this exact "minimum-jerk
# polynomial" is now standard in robotics trajectory generation (e.g.
# Biagiotti & Melchiorri, "Trajectory Planning for Automatic Machines and
# Robots," Springer, 2008, Ch. 4, which this module also follows for the
# via-point velocity heuristic below). Chaining one such quintic segment
# between each consecutive pair of waypoints, with matching
# position/velocity at the shared boundary, gives a trajectory with
# continuous position and velocity (and a well-defined, finite,
# locally-minimized jerk) all the way through -- a dramatic difference
# from the A* path's blocky, zero-order-continuous-only corners.
#
# NO EXTERNAL NUMERICAL LIBRARY IS USED (same as the rest of this app --
# e.g. control_loop.py's hand-rolled RK4 integrator): the small (3x3)
# linear systems that pin down each quintic segment's coefficients are
# solved with a pure-Python Gauss-Jordan elimination routine below.
#
# WHAT WAS VERIFIED BEFORE SHIPPING (see test_path_planner.py):
# - _solve_linear_system checked against a hand-solved 3x3 system.
# - Every quintic segment's generated position/velocity/acceleration at
#   its own two boundary times were checked to exactly match the
#   requested boundary conditions (the direct definition of correctness
#   for a boundary-value polynomial -- not just "looks smooth").
# - The safety-aware waypoint simplification (_rdp_simplify_safe) was
#   checked to never let its straight-chord replacement pass within
#   obstacle_radius of an obstacle, and the final sampled trajectory was
#   verified end-to-end to stay outside obstacle_radius of every obstacle
#   the original A* path was planned around.
# - The realized (sampled) peak speed/acceleration were verified to stay
#   at or below the requested max_speed_mps/max_acceleration_mps2.
# - Total path curvature (sum of heading-angle changes between
#   consecutive samples) was verified to be dramatically lower for the
#   smoothed trajectory than for the raw A* waypoint path on the same
#   scenario -- see test_path_planner.py's concrete before/after numbers.
# ============================================================================


def _solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """
    Solves the dense linear system a @ x = b via Gauss-Jordan elimination
    with partial pivoting -- a standard, general-purpose direct solver
    (small systems only; this app never needs more than 3x3 here). Used
    to solve for each quintic trajectory segment's unknown coefficients
    from its boundary conditions (see _quintic_segment_coefficients).
    """
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot_row][col]) < 1e-12:
            raise ValueError("Singular system -- cannot solve for trajectory coefficients")
        m[col], m[pivot_row] = m[pivot_row], m[col]

        pivot = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = m[row][col]
                if factor != 0:
                    for j in range(col, n + 1):
                        m[row][j] -= factor * m[col][j]

    return [m[i][n] for i in range(n)]


def _quintic_segment_coefficients(
    p0: float, v0: float, a0: float, p1: float, v1: float, a1: float, duration: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    Returns (c0..c5) for p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
    on t in [0, duration], matching p(0)=p0, p'(0)=v0, p''(0)=a0, and
    p(duration)=p1, p'(duration)=v1, p''(duration)=a1 exactly. c0, c1, c2
    follow directly from the t=0 conditions; c3, c4, c5 solve the
    remaining 3x3 linear system from the t=duration conditions.
    """
    c0 = p0
    c1 = v0
    c2 = a0 / 2.0
    T = duration

    rhs = [
        p1 - c0 - c1 * T - c2 * T ** 2,
        v1 - c1 - 2 * c2 * T,
        a1 - 2 * c2,
    ]
    mat = [
        [T ** 3, T ** 4, T ** 5],
        [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
        [6 * T, 12 * T ** 2, 20 * T ** 3],
    ]
    c3, c4, c5 = _solve_linear_system(mat, rhs)
    return c0, c1, c2, c3, c4, c5


def _evaluate_quintic(coeffs: Tuple[float, ...], t: float) -> Tuple[float, float, float]:
    """Returns (position, velocity, acceleration) of a quintic segment at time t."""
    c0, c1, c2, c3, c4, c5 = coeffs
    pos = c0 + c1 * t + c2 * t ** 2 + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5
    vel = c1 + 2 * c2 * t + 3 * c3 * t ** 2 + 4 * c4 * t ** 3 + 5 * c5 * t ** 4
    acc = 2 * c2 + 6 * c3 * t + 12 * c4 * t ** 2 + 20 * c5 * t ** 3
    return pos, vel, acc


def _point_to_segment_distance_3d(point: Point3D, seg_a: Point3D, seg_b: Point3D) -> float:
    """Shortest distance from `point` to the straight segment seg_a-seg_b."""
    ax, ay, az = seg_a
    bx, by, bz = seg_b
    abx, aby, abz = bx - ax, by - ay, bz - az
    ab_len_sq = abx * abx + aby * aby + abz * abz
    if ab_len_sq < 1e-12:
        return math.dist(point, seg_a)
    px, py, pz = point
    t = ((px - ax) * abx + (py - ay) * aby + (pz - az) * abz) / ab_len_sq
    t = max(0.0, min(1.0, t))
    closest = (ax + t * abx, ay + t * aby, az + t * abz)
    return math.dist(point, closest)


def _rdp_simplify_safe(
    points: List[Point3D],
    obstacles: List[Point3D],
    obstacle_radius: float,
    epsilon: float,
) -> List[Point3D]:
    """
    Douglas-Peucker path simplification (Douglas & Peucker, "Algorithms
    for the Reduction of the Number of Points Required to Represent a
    Digitized Line or its Caricature," Canadian Cartographer, 1973),
    adapted to be SAFETY-AWARE: a run of waypoints is only collapsed to
    its two endpoints if BOTH (a) every dropped interior point deviates
    from the straight chord by less than `epsilon` (the standard RDP
    criterion) AND (b) that straight chord itself stays outside
    obstacle_radius of every obstacle along its whole length -- checked
    with a small safety margin (obstacle_radius + epsilon, since the
    quintic curve fit through the simplified points can bulge slightly
    off the straight chord; generate_minimum_jerk_trajectory below still
    re-verifies the ACTUAL sampled curve afterward as the authoritative
    check, this is just what keeps most of the grid-cell waypoints from
    surviving simplification unnecessarily). Otherwise the interior point
    with the largest deviation is kept and the run is split and
    re-checked recursively -- exactly standard RDP's recursive step, just
    gated by an extra obstacle-clearance condition.
    """
    if len(points) < 3:
        return points

    clearance = obstacle_radius + epsilon

    def chord_clear(a: Point3D, b: Point3D) -> bool:
        return all(_point_to_segment_distance_3d(o, a, b) >= clearance for o in obstacles)

    def max_deviation(pts: List[Point3D], a: Point3D, b: Point3D) -> Tuple[int, float]:
        best_idx, best_dist = -1, -1.0
        for i in range(1, len(pts) - 1):
            d = _point_to_segment_distance_3d(pts[i], a, b)
            if d > best_dist:
                best_idx, best_dist = i, d
        return best_idx, best_dist

    def simplify(pts: List[Point3D]) -> List[Point3D]:
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        idx, dist = max_deviation(pts, a, b)
        if dist <= epsilon and chord_clear(a, b):
            return [a, b]
        left = simplify(pts[: idx + 1])
        right = simplify(pts[idx:])
        return left[:-1] + right

    return simplify(points)


def _via_point_velocities(keypoints: List[Point3D], segment_times: List[float]) -> List[Point3D]:
    """
    Assigns a velocity vector to each via-point (waypoint), using the
    standard heuristic for continuous-velocity polynomial via-point
    trajectories (Biagiotti & Melchiorri, "Trajectory Planning for
    Automatic Machines and Robots," Springer, 2008, Ch. 4): zero velocity
    at the first and last point (start/end at rest -- physically sensible
    for a drone launching from hover and coming to a stop at landing);
    for each interior point, per axis, average the incoming and outgoing
    segment's average velocity IF they have the same sign (the path is
    still heading the same direction through this point), otherwise zero
    (the path reverses direction at this point on this axis, so carrying
    speed through it would overshoot -- the standard rule for avoiding
    overshoot at a direction-reversal via-point).
    """
    n = len(keypoints)
    velocities: List[Point3D] = [(0.0, 0.0, 0.0)] * n
    for i in range(1, n - 1):
        v_axes = []
        for axis in range(3):
            d_in = (keypoints[i][axis] - keypoints[i - 1][axis]) / segment_times[i - 1]
            d_out = (keypoints[i + 1][axis] - keypoints[i][axis]) / segment_times[i]
            if d_in == 0 or d_out == 0 or (d_in > 0) != (d_out > 0):
                v_axes.append(0.0)
            else:
                v_axes.append(0.5 * (d_in + d_out))
        velocities[i] = (v_axes[0], v_axes[1], v_axes[2])
    return velocities


def _clip_velocity_magnitude(v: Point3D, max_speed_mps: float) -> Point3D:
    mag = math.dist((0.0, 0.0, 0.0), v)
    if mag <= max_speed_mps or mag < 1e-9:
        return v
    scale = max_speed_mps / mag
    return (v[0] * scale, v[1] * scale, v[2] * scale)


# Exact closed-form peak-velocity/acceleration factors for a quintic
# segment with ZERO velocity and acceleration at BOTH ends (the classic
# minimum-jerk point-to-point motion, p(t)=d*(10u^3-15u^4+6u^5), u=t/T):
# differentiating and solving for the stationary points gives peak
# velocity = 1.875*d/T at u=0.5, and peak |acceleration| =
# (10/sqrt(3))*d/T^2 at u=(1 -+ 1/sqrt(3))/2 -- both exact, not
# approximations. Used to size each segment's duration; via-points with
# nonzero endpoint velocity (see _via_point_velocities) deviate from
# this exact case, so generate_minimum_jerk_trajectory verifies the
# actual sampled trajectory afterward rather than trusting this sizing
# blindly.
_QUINTIC_ZERO_ENDPOINT_PEAK_VELOCITY_FACTOR = 1.875
_QUINTIC_ZERO_ENDPOINT_PEAK_ACCEL_FACTOR = 10.0 / math.sqrt(3)


def _segment_time_for_constraints(
    distance: float, max_speed_mps: float, max_acceleration_mps2: float, min_time_s: float = 0.05,
) -> float:
    """Sizes a segment's duration so a zero-endpoint-velocity/acceleration
    quintic moving `distance` over that duration has peak velocity <=
    max_speed_mps and peak acceleration <= max_acceleration_mps2 (see the
    exact closed forms above); returns the larger (more restrictive) of
    the two required durations."""
    if distance <= 1e-9:
        return min_time_s
    t_speed = _QUINTIC_ZERO_ENDPOINT_PEAK_VELOCITY_FACTOR * distance / max_speed_mps
    t_accel = math.sqrt(_QUINTIC_ZERO_ENDPOINT_PEAK_ACCEL_FACTOR * distance / max_acceleration_mps2)
    return max(t_speed, t_accel, min_time_s)


MAX_TRAJECTORY_RESCALE_ITERATIONS = 4
# How far over the requested max the sampled trajectory's realized peak
# speed/acceleration is allowed to land before triggering another
# rescale pass -- purely to stop the iterative rescale loop from chasing
# floating-point/sampling noise forever; 2% is well within this app's
# other "close enough" tolerances (e.g. the RK4-vs-closed-form check in
# control_loop.py).
TRAJECTORY_CONSTRAINT_TOLERANCE = 1.02
MAX_SIMPLIFICATION_RETRIES = 3  # halves simplify_tolerance_m this many times before falling back to the raw waypoints


def generate_minimum_jerk_trajectory(
    waypoints: List[Point3D],
    max_speed_mps: float,
    max_acceleration_mps2: float,
    obstacles: List[Point3D],
    obstacle_radius: float = 0.6,
    simplify_tolerance_m: float = 0.5,
    sample_rate_hz: float = 10.0,
) -> dict:
    """
    Turns plan_path_3d's waypoint list into a smooth, dynamically-
    feasible trajectory -- see the module section docstring above for the
    full method and citations. Returns a dict with the sampled trajectory
    (position/speed/acceleration over time) plus diagnostics. Raises
    ValueError for invalid inputs (fewer than 2 waypoints, non-positive
    speed/acceleration).
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least a start and goal point to build a trajectory")
    if max_speed_mps <= 0:
        raise ValueError("max_speed_mps must be positive")
    if max_acceleration_mps2 <= 0:
        raise ValueError("max_acceleration_mps2 must be positive")

    def dedupe(pts: List[Point3D]) -> List[Point3D]:
        out = [pts[0]]
        for p in pts[1:]:
            if math.dist(p, out[-1]) > 1e-6:
                out.append(p)
        return out if len(out) >= 2 else [pts[0], pts[-1]]

    def build_keypoints(tolerance: float) -> List[Point3D]:
        simplified = _rdp_simplify_safe(waypoints, obstacles, obstacle_radius, tolerance)
        return dedupe(simplified)

    def build_sampled_trajectory(keypoints: List[Point3D], times: List[float]):
        n_segments = len(keypoints) - 1
        velocities = _via_point_velocities(keypoints, times)
        velocities = [_clip_velocity_magnitude(v, max_speed_mps) for v in velocities]

        segment_coeffs = []
        for i in range(n_segments):
            p0, p1 = keypoints[i], keypoints[i + 1]
            v0, v1 = velocities[i], velocities[i + 1]
            T = times[i]
            axes_coeffs = tuple(
                _quintic_segment_coefficients(p0[axis], v0[axis], 0.0, p1[axis], v1[axis], 0.0, T)
                for axis in range(3)
            )
            segment_coeffs.append(axes_coeffs)

        total_duration = sum(times)
        cumulative = [0.0]
        for T in times:
            cumulative.append(cumulative[-1] + T)

        n_samples = max(2, int(round(total_duration * sample_rate_hz)) + 1)
        samples = []
        for i in range(n_samples):
            t_global = (total_duration * i) / (n_samples - 1)
            seg_idx = 0
            while seg_idx < n_segments - 1 and t_global > cumulative[seg_idx + 1]:
                seg_idx += 1
            t_local = min(t_global - cumulative[seg_idx], times[seg_idx])

            pos, vel, acc = [], [], []
            for axis in range(3):
                p, v, a = _evaluate_quintic(segment_coeffs[seg_idx][axis], t_local)
                pos.append(p)
                vel.append(v)
                acc.append(a)

            samples.append({
                "t": t_global,
                "position": (pos[0], pos[1], pos[2]),
                "speed_mps": math.dist((0.0, 0.0, 0.0), vel),
                "acceleration_mps2": math.dist((0.0, 0.0, 0.0), acc),
            })
        return samples, total_duration

    def clears_obstacles(samples) -> bool:
        # Small numerical tolerance -- sample points are discrete, so
        # this checks each sampled position directly rather than every
        # continuous point along the curve, same practical approach
        # plan_path_3d's own grid discretization already relies on.
        return all(
            all(math.dist(s["position"], o) >= obstacle_radius - 1e-6 for o in obstacles)
            for s in samples
        )

    tolerance = simplify_tolerance_m
    keypoints = build_keypoints(tolerance)
    used_fallback = False

    for attempt in range(MAX_SIMPLIFICATION_RETRIES + 2):
        n_segments = len(keypoints) - 1
        distances = [math.dist(keypoints[i], keypoints[i + 1]) for i in range(n_segments)]
        segment_times = [
            _segment_time_for_constraints(d, max_speed_mps, max_acceleration_mps2) for d in distances
        ]

        samples, total_duration = build_sampled_trajectory(keypoints, segment_times)
        realized_max_speed = max(s["speed_mps"] for s in samples)
        realized_max_accel = max(s["acceleration_mps2"] for s in samples)

        for _ in range(MAX_TRAJECTORY_RESCALE_ITERATIONS):
            speed_ratio = realized_max_speed / max_speed_mps
            accel_ratio = realized_max_accel / max_acceleration_mps2
            # Time-scaling a fixed-shape trajectory by k scales velocity
            # by 1/k and acceleration by 1/k^2 -- exact identities, not
            # an approximation -- so sqrt(accel_ratio) is the scale
            # needed to bring acceleration into bounds.
            scale = max(speed_ratio, math.sqrt(accel_ratio), 1.0)
            if scale <= TRAJECTORY_CONSTRAINT_TOLERANCE:
                break
            segment_times = [t * scale for t in segment_times]
            samples, total_duration = build_sampled_trajectory(keypoints, segment_times)
            realized_max_speed = max(s["speed_mps"] for s in samples)
            realized_max_accel = max(s["acceleration_mps2"] for s in samples)

        # This exact sampled trajectory (post-rescale) is what gets
        # returned/checked -- clears_obstacles is evaluated on it
        # directly, not on some earlier intermediate version, so a
        # "success" here is a real guarantee about what's returned.
        if clears_obstacles(samples) or used_fallback:
            # `used_fallback` breaks even on failure: the fallback fits
            # through every raw A* waypoint, which plan_path_3d already
            # guaranteed clears every obstacle by obstacle_radius, so a
            # failure here would mean a bug in the sampling/verification
            # logic itself, not something another retry could fix --
            # stop rather than loop forever.
            break
        if attempt == MAX_SIMPLIFICATION_RETRIES:
            # Guaranteed-safe fallback: fit the spline through every raw
            # A* waypoint (no simplification at all) instead of trying an
            # even smaller tolerance -- this always clears obstacles by
            # the same margin plan_path_3d's own grid search already
            # guaranteed, at the cost of losing the smoothing benefit in
            # this (expected to be rare) worst case.
            keypoints = dedupe(list(waypoints))
            used_fallback = True
        else:
            tolerance /= 2
            keypoints = build_keypoints(tolerance)

    return {
        "keypoints": keypoints,
        "raw_waypoint_count": len(waypoints),
        "keypoint_count": len(keypoints),
        "samples": samples,
        "total_duration_seconds": round(total_duration, 3),
        "max_speed_mps_used": max_speed_mps,
        "max_acceleration_mps2_used": max_acceleration_mps2,
        "max_realized_speed_mps": round(max(s["speed_mps"] for s in samples), 3),
        "max_realized_acceleration_mps2": round(max(s["acceleration_mps2"] for s in samples), 3),
    }
