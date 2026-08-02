"""
Standalone test script for the 3D path planner extension -- run directly
with `python3 test_path_planner_3d.py`, matching the other standalone
test_*.py scripts in this repo.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

from app.services.path_planner import (
    plan_path, path_distance_meters,
    plan_path_3d, path_distance_meters_3d,
)


def test_clear_path_is_roughly_straight_line():
    path = plan_path_3d(start=(0, 0, 0), goal=(10, 0, 0), obstacles=[])
    assert path is not None
    assert path[0] == (0, 0, 0)

    straight_line_distance = 10.0
    actual_distance = path_distance_meters_3d(path)
    assert actual_distance < straight_line_distance * 1.05, (
        f"Clear path should be nearly straight; got {actual_distance:.2f}m vs {straight_line_distance}m"
    )
    print(f"PASS: clear 3D path, length={actual_distance:.2f}m (straight-line={straight_line_distance}m)")


def test_path_routes_around_a_single_obstacle_by_climbing_over_it():
    # Obstacle blocks the direct route at the SAME altitude as start/goal
    # -- a real 3D planner should be able to route around it either
    # laterally OR by climbing over/under it (something the 2D planner
    # could never do, since it has no altitude axis at all).
    obstacle = (5, 0, 0)
    obstacle_radius = 0.6
    path = plan_path_3d(start=(0, 0, 0), goal=(10, 0, 0), obstacles=[obstacle], obstacle_radius=obstacle_radius)
    assert path is not None, "Expected a path that routes around the 3D obstacle"

    straight_line_distance = 10.0
    actual_distance = path_distance_meters_3d(path)
    assert actual_distance > straight_line_distance, "Path around an obstacle should be longer than the direct line"

    for x, y, z in path:
        dist_to_obstacle = math.dist((x, y, z), obstacle)
        assert dist_to_obstacle >= obstacle_radius - 0.01, (
            f"Path point ({x:.2f},{y:.2f},{z:.2f}) is inside the obstacle's safety radius"
        )
    print(f"PASS: routes around 3D obstacle, length={actual_distance:.2f}m (straight-line={straight_line_distance}m)")


def test_fully_enclosed_goal_returns_none():
    # A sphere-shell "cage" of obstacles fully surrounding the goal --
    # no path should exist in any of the 3 dimensions.
    goal = (10, 5, 10)
    ring_radius = 2.5
    obstacle_radius = 1.0
    cage = []
    # Two rings at different "latitudes" of a sphere approximate a sealed
    # shell closely enough for obstacle_radius=1.0 spacing to overlap.
    for lat_deg, num in [(-45, 10), (0, 14), (45, 10)]:
        lat = math.radians(lat_deg)
        for i in range(num):
            lon = 2 * math.pi * i / num
            x = goal[0] + ring_radius * math.cos(lat) * math.cos(lon)
            y = goal[1] + ring_radius * math.sin(lat)
            z = goal[2] + ring_radius * math.cos(lat) * math.sin(lon)
            cage.append((x, y, z))
    # Cap the very top and bottom too.
    cage.append((goal[0], goal[1] + ring_radius, goal[2]))
    cage.append((goal[0], goal[1] - ring_radius, goal[2]))

    path = plan_path_3d(start=(0, 5, 0), goal=goal, obstacles=cage, obstacle_radius=obstacle_radius)
    assert path is None, "Expected no path when the goal is fully sealed inside a 3D cage of obstacles"
    print("PASS: fully enclosed 3D goal correctly returns None (no path)")


def test_start_inside_obstacle_raises_clear_error():
    try:
        plan_path_3d(start=(5, 2, 5), goal=(10, 2, 10), obstacles=[(5, 2, 5)], obstacle_radius=1.0)
        assert False, "Expected a ValueError when start is inside an obstacle"
    except ValueError as e:
        assert "Start" in str(e)
        print(f"PASS: start-inside-obstacle raises clear error: {e}")


def test_flat_altitude_scenario_is_consistent_with_the_2d_planner():
    # Regression-safety cross-check: when every obstacle/start/goal sits
    # at the SAME altitude (y=0), the 3D planner has no reason to use its
    # extra dimension and should find a path of essentially the same
    # length as the existing, already-tested 2D planner for the
    # equivalent (x, z) layout -- confirms the 3D generalization didn't
    # change behavior for the case the 2D planner already handles.
    obstacle_2d = (5, 0)
    obstacle_3d = (5, 0, 0)
    obstacle_radius = 0.6

    path_2d = plan_path(start=(0, 0), goal=(10, 0), obstacles=[obstacle_2d], obstacle_radius=obstacle_radius)
    path_3d = plan_path_3d(start=(0, 0, 0), goal=(10, 0, 0), obstacles=[obstacle_3d], obstacle_radius=obstacle_radius)

    dist_2d = path_distance_meters(path_2d)
    dist_3d = path_distance_meters_3d(path_3d)

    assert abs(dist_2d - dist_3d) < 0.5, (
        f"Expected 2D and 3D planners to agree closely on a flat-altitude layout, "
        f"got 2D={dist_2d:.2f}m vs 3D={dist_3d:.2f}m"
    )
    print(f"PASS: flat-altitude 3D result ({dist_3d:.2f}m) consistent with 2D planner ({dist_2d:.2f}m)")


if __name__ == "__main__":
    test_clear_path_is_roughly_straight_line()
    test_path_routes_around_a_single_obstacle_by_climbing_over_it()
    test_fully_enclosed_goal_returns_none()
    test_start_inside_obstacle_raises_clear_error()
    test_flat_altitude_scenario_is_consistent_with_the_2d_planner()
    print("\nAll 3D path planner tests passed.")
