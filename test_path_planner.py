"""
Standalone test script for the path planner -- run directly with
`python3 test_path_planner.py`, no pytest/framework needed, so it can run
anywhere without extra setup.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.path_planner import plan_path, path_distance_meters


def test_clear_path_is_roughly_straight_line():
    """No obstacles between start and goal -- path should be short and
    close to the direct straight-line distance (allowing a little grid
    quantization error, not an exact match)."""
    path = plan_path(start=(0, 0), goal=(10, 0), obstacles=[])
    assert path is not None, "Expected a path with no obstacles present"
    assert path[0] == (0, 0), f"Path should start at start point, got {path[0]}"

    straight_line_distance = 10.0
    actual_distance = path_distance_meters(path)
    assert actual_distance < straight_line_distance * 1.05, (
        f"Clear path should be nearly straight; got {actual_distance:.2f}m "
        f"vs straight-line {straight_line_distance}m"
    )
    print(f"PASS: clear path, length={actual_distance:.2f}m (straight-line={straight_line_distance}m)")


def test_path_routes_around_a_single_obstacle():
    """One obstacle directly in the middle of the straight line -- the
    path must be LONGER than a straight line (it has to go around) and
    must not pass through the obstacle's safety radius."""
    obstacle = (5, 0)
    obstacle_radius = 0.6
    path = plan_path(start=(0, 0), goal=(10, 0), obstacles=[obstacle], obstacle_radius=obstacle_radius)
    assert path is not None, "Expected a path that routes around the obstacle"

    straight_line_distance = 10.0
    actual_distance = path_distance_meters(path)
    assert actual_distance > straight_line_distance, (
        "Path around an obstacle should be longer than the direct straight line"
    )

    import math
    for x, z in path:
        dist_to_obstacle = math.hypot(x - obstacle[0], z - obstacle[1])
        assert dist_to_obstacle >= obstacle_radius - 0.01, (
            f"Path point ({x:.2f}, {z:.2f}) is inside the obstacle's safety radius "
            f"(distance={dist_to_obstacle:.2f}, required>={obstacle_radius})"
        )
    print(f"PASS: routes around obstacle, length={actual_distance:.2f}m (straight-line={straight_line_distance}m), all points clear of obstacle")


def test_fully_enclosed_goal_returns_none():
    """Goal is surrounded on all sides by a sealed ring of obstacles -- no
    path should exist, and the function should say so (None) rather than
    silently returning a path that cuts through an obstacle.

    The ring is placed far enough from the goal that the goal point ITSELF
    isn't blocked (that would raise a different, more specific error --
    see test_start_inside_obstacle_raises_clear_error below for that case)
    -- this test is specifically about a goal that's reachable in principle
    but has no CLEAR PATH to it.
    """
    import math

    goal = (10, 10)
    ring_radius = 2.5
    obstacle_radius = 1.0
    num_obstacles = 16  # closely spaced enough that their radii overlap, sealing every gap

    ring = []
    for i in range(num_obstacles):
        angle = 2 * math.pi * i / num_obstacles
        ring.append((goal[0] + ring_radius * math.cos(angle), goal[1] + ring_radius * math.sin(angle)))

    path = plan_path(start=(0, 0), goal=goal, obstacles=ring, obstacle_radius=obstacle_radius)
    assert path is None, "Expected no path when the goal is fully sealed off by a ring of obstacles"
    print("PASS: fully enclosed goal correctly returns None (no path)")


def test_start_inside_obstacle_raises_clear_error():
    """Starting position itself is inside an obstacle -- should raise a
    clear, specific error rather than a confusing crash or silent bad path."""
    try:
        plan_path(start=(5, 5), goal=(10, 10), obstacles=[(5, 5)], obstacle_radius=1.0)
        assert False, "Expected a ValueError when start is inside an obstacle"
    except ValueError as e:
        assert "Start" in str(e)
        print(f"PASS: start-inside-obstacle raises clear error: {e}")


if __name__ == "__main__":
    test_clear_path_is_roughly_straight_line()
    test_path_routes_around_a_single_obstacle()
    test_fully_enclosed_goal_returns_none()
    test_start_inside_obstacle_raises_clear_error()
    print("\nAll path planner tests passed.")
