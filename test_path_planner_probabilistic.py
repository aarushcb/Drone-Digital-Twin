"""
Standalone test script for probabilistic/risk-aware 3D path planning --
run directly with `python3 test_path_planner_probabilistic.py`, matching
the other standalone test_*.py scripts in this repo.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

from app.services.path_planner import (
    plan_path_3d, path_distance_meters_3d,
    plan_path_3d_probabilistic, collision_probability, _sample_obstacle_positions,
)


def test_collision_probability_near_1_at_obstacle_mean_near_0_far_away():
    # safety_radius = 3 * std here (~"3-sigma" in each axis) -- for an
    # isotropic 3D Gaussian, the radial distance from the mean follows a
    # chi distribution with 3 degrees of freedom, whose CDF at 3 puts
    # about 97% of probability mass within that radius (a standard result
    # for the chi-3/Maxwell distribution) -- used here just to pick test
    # parameters that produce an unambiguously "near 1" probability,
    # not as a claim the code computes this analytically (it estimates
    # it via Monte Carlo, same as the rest of this function).
    std = 0.3
    samples = _sample_obstacle_positions([(5, 0, 0)], obstacle_position_std=std, num_samples=3000, seed=1)
    at_mean = collision_probability((5, 0, 0), samples, safety_radius=3 * std)
    far_away = collision_probability((50, 0, 0), samples, safety_radius=3 * std)
    assert at_mean > 0.9, f"Expected near-1 collision probability at the obstacle's own mean (radius=3*std), got {at_mean}"
    assert far_away < 0.01, f"Expected near-0 collision probability far from the obstacle, got {far_away}"
    print(f"PASS: collision probability at obstacle mean (radius=3*std)={at_mean:.3f}, far away={far_away:.4f}")


def test_no_obstacles_gives_a_clear_straight_path_with_zero_risk():
    result = plan_path_3d_probabilistic(
        start=(0, 0, 0), goal=(10, 0, 0), obstacle_means=[], obstacle_position_std=0.5, seed=1,
    )
    assert result is not None
    path, risks = result
    assert path[0] == (0, 0, 0)
    distance = path_distance_meters_3d(path)
    assert distance < 10.5, f"Expected a nearly-straight path with no obstacles, got {distance:.2f}m"
    assert all(r == 0.0 for r in risks), f"Expected zero collision risk with no obstacles, got {risks}"
    print(f"PASS: no-obstacle path is clear (length={distance:.2f}m) with zero risk along its whole length")


def test_zero_uncertainty_reduces_to_the_hard_radius_planner():
    # At obstacle_position_std=0, every Monte Carlo sample lands exactly
    # on the obstacle's mean position -- collision_probability becomes a
    # step function equivalent to plan_path_3d's hard obstacle_radius
    # check. The two planners should find paths of very similar length
    # in this limit -- the same kind of reduction check already used for
    # plan_path_3d's flat-altitude-vs-2D consistency test.
    obstacle = (5, 0, 0)
    safety_radius = 0.6

    hard_path = plan_path_3d(start=(0, 0, 0), goal=(10, 0, 0), obstacles=[obstacle], obstacle_radius=safety_radius)
    hard_distance = path_distance_meters_3d(hard_path)

    prob_result = plan_path_3d_probabilistic(
        start=(0, 0, 0), goal=(10, 0, 0), obstacle_means=[obstacle],
        obstacle_position_std=0.0, safety_radius=safety_radius, seed=1,
    )
    assert prob_result is not None
    prob_path, _ = prob_result
    prob_distance = path_distance_meters_3d(prob_path)

    assert abs(hard_distance - prob_distance) < 1.0, (
        f"Expected zero-uncertainty probabilistic planner to closely match the hard-radius planner, "
        f"got hard={hard_distance:.2f}m vs probabilistic={prob_distance:.2f}m"
    )
    print(f"PASS: zero-uncertainty probabilistic result ({prob_distance:.2f}m) matches hard-radius planner ({hard_distance:.2f}m)")


def test_more_obstacle_uncertainty_forces_a_wider_detour():
    # The core, physically-meaningful behavior this feature exists for:
    # more positional uncertainty about an obstacle should force the
    # planner to give it a wider berth, producing a longer path -- not
    # just "some numbers changed," a specific, testable, expected direction.
    obstacle = (5, 0, 0)
    tight = plan_path_3d_probabilistic(
        start=(0, 0, 0), goal=(10, 0, 0), obstacle_means=[obstacle],
        obstacle_position_std=0.1, safety_radius=0.6, seed=42,
    )
    loose = plan_path_3d_probabilistic(
        start=(0, 0, 0), goal=(10, 0, 0), obstacle_means=[obstacle],
        obstacle_position_std=1.5, safety_radius=0.6, seed=42,
    )
    assert tight is not None and loose is not None
    tight_distance = path_distance_meters_3d(tight[0])
    loose_distance = path_distance_meters_3d(loose[0])

    assert loose_distance > tight_distance, (
        f"Expected more obstacle position uncertainty to force a longer detour, "
        f"got tight(std=0.1)={tight_distance:.2f}m vs loose(std=1.5)={loose_distance:.2f}m"
    )
    print(
        f"PASS: detour grows with obstacle uncertainty -- "
        f"std=0.1 -> {tight_distance:.2f}m, std=1.5 -> {loose_distance:.2f}m"
    )


def test_path_avoids_near_certain_collision_cells():
    # The path returned should never pass through a cell whose collision
    # probability exceeds the hard-block threshold -- confirms the
    # "still treat near-certain collision as impassable" safety backstop
    # actually works, not just that risk is *penalized*.
    from app.services.path_planner import HARD_BLOCK_THRESHOLD, _sample_obstacle_positions, collision_probability

    obstacle = (5, 0, 0)
    obstacle_position_std = 0.3
    safety_radius = 0.6
    result = plan_path_3d_probabilistic(
        start=(0, 0, 0), goal=(10, 0, 0), obstacle_means=[obstacle],
        obstacle_position_std=obstacle_position_std, safety_radius=safety_radius, seed=5,
    )
    assert result is not None
    path, risks = result
    assert all(r <= HARD_BLOCK_THRESHOLD for r in risks), f"Path passes through a near-certain-collision cell: {risks}"
    print(f"PASS: entire path stays at or below the hard-block threshold ({HARD_BLOCK_THRESHOLD}) -- max risk on path = {max(risks):.3f}")


def test_default_call_is_deterministic_across_repeated_calls():
    # A path planner's output shouldn't vary between two identical
    # requests -- unlike monte_carlo_uq.py's functions (which report
    # aggregate statistics that are stable across different seeds by
    # design), this feeds the Monte Carlo estimate directly into which
    # literal path is returned, so it defaults to a FIXED seed rather
    # than None. Confirms that default actually produces identical
    # results across repeated calls with no seed passed explicitly.
    kwargs = dict(
        start=(0, 0, 0), goal=(10, 0, 0), obstacle_means=[(5, 0, 0)], obstacle_position_std=0.8,
    )
    results = [plan_path_3d_probabilistic(**kwargs) for _ in range(4)]
    distances = [path_distance_meters_3d(r[0]) for r in results]
    assert len(set(distances)) == 1, f"Expected identical results across repeated calls with no seed override, got {distances}"
    print(f"PASS: repeated calls with no explicit seed are deterministic -- all 4 gave distance={distances[0]:.2f}m")


def test_start_inside_near_certain_collision_raises_clear_error():
    try:
        plan_path_3d_probabilistic(
            start=(5, 0, 0), goal=(10, 10, 10), obstacle_means=[(5, 0, 0)],
            obstacle_position_std=0.05, safety_radius=1.0, seed=1,
        )
        assert False, "Expected a ValueError when start has near-certain collision probability"
    except ValueError as e:
        assert "Start" in str(e)
        print(f"PASS: start-in-near-certain-collision raises clear error: {e}")


if __name__ == "__main__":
    test_collision_probability_near_1_at_obstacle_mean_near_0_far_away()
    test_no_obstacles_gives_a_clear_straight_path_with_zero_risk()
    test_zero_uncertainty_reduces_to_the_hard_radius_planner()
    test_more_obstacle_uncertainty_forces_a_wider_detour()
    test_path_avoids_near_certain_collision_cells()
    test_default_call_is_deterministic_across_repeated_calls()
    test_start_inside_near_certain_collision_raises_clear_error()
    print("\nAll probabilistic path planner tests passed.")
