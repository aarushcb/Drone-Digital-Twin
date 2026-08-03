"""
Standalone test script for the forward-flight / wind extension to BEMT --
run directly with `python3 test_bemt_wind.py`, matching the other
standalone test_*.py scripts in this repo.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.bemt import (
    glauert_inflow_ratio, bemt_thrust_and_power, bemt_thrust_and_power_forward_flight,
    analyze_bemt_hover_in_wind, hover_coefficients,
)

SEA_LEVEL_DENSITY = 1.225


def test_zero_wind_reduces_exactly_to_hover():
    # The core correctness check: this module claims to be a strict
    # generalization of the hover-only model, not a different model that
    # coincidentally agrees at mu=0. At wind_speed_mps=0, everything
    # should match the existing hover BEMT function to floating-point
    # precision.
    diameter_m = 10 * 0.0254
    rpm = 5000
    hover = bemt_thrust_and_power(diameter_m, rpm, SEA_LEVEL_DENSITY)
    zero_wind = bemt_thrust_and_power_forward_flight(diameter_m, rpm, SEA_LEVEL_DENSITY, wind_speed_mps=0)

    assert abs(hover["ct"] - zero_wind["ct"]) < 1e-9
    assert abs(hover["thrust_n"] - zero_wind["thrust_n"]) < 1e-6
    assert abs(hover["power_w"] - zero_wind["power_w"]) < 1e-4, (
        f"hover power={hover['power_w']}, zero-wind power={zero_wind['power_w']}"
    )
    assert zero_wind["advance_ratio"] == 0.0
    print(
        f"PASS: wind_speed=0 reproduces hover exactly "
        f"(hover power={hover['power_w']:.3f}W, forward-flight-model-at-mu=0 power={zero_wind['power_w']:.3f}W)"
    )


def test_glauert_inflow_matches_hover_at_zero_advance_ratio():
    # At mu=0, Glauert's equation must reduce to the well-known hover
    # closed form lambda_h = sqrt(CT/2) -- an independent algebraic check
    # of the iterative solver, not just "the wrapper function agrees."
    ct = 0.0104
    lam = glauert_inflow_ratio(ct, advance_ratio=0.0)
    expected = (ct / 2) ** 0.5
    assert abs(lam - expected) < 1e-6, f"Expected lambda={expected}, got {lam}"
    print(f"PASS: Glauert inflow at mu=0 matches hover closed form (lambda={lam:.5f}, expected={expected:.5f})")


def test_power_follows_the_classic_bucket_curve_shape():
    # NOT a naive "power always goes up in wind" check -- real rotorcraft
    # aerodynamics (Leishman Ch. 5's classic "power required vs. forward
    # speed" curve) predicts power INITIALLY DROPS as wind/forward speed
    # increases from hover (translational lift reduces induced power),
    # reaches a minimum, then rises again once profile drag (which grows
    # as mu^2) takes over. This test confirms this model reproduces that
    # real, well-documented, counterintuitive-sounding shape rather than
    # naively increasing monotonically with wind speed.
    diameter_m = 10 * 0.0254
    rpm = 5000
    wind_speeds = [0, 3, 6, 10, 15, 20, 25, 30, 35, 40]
    powers = [
        bemt_thrust_and_power_forward_flight(diameter_m, rpm, SEA_LEVEL_DENSITY, w)["power_w"]
        for w in wind_speeds
    ]
    min_power = min(powers)
    min_idx = powers.index(min_power)

    assert powers[0] == max(powers[:len(powers) // 2 + 1]), "Expected power to start dropping from the hover value"
    assert powers[1] < powers[0], "Expected power to decrease immediately past hover (translational lift)"
    assert 0 < min_idx < len(powers) - 1, f"Expected an interior minimum (bucket curve), got minimum at index {min_idx}"
    assert powers[-1] > min_power, "Expected power to rise again at high wind speed (profile drag dominates)"
    print(
        f"PASS: bucket-curve shape confirmed -- power drops from {powers[0]:.2f}W at hover to a minimum "
        f"of {min_power:.2f}W at {wind_speeds[min_idx]}m/s, then rises to {powers[-1]:.2f}W at "
        f"{wind_speeds[-1]}m/s -> {list(zip(wind_speeds, [round(p,2) for p in powers]))}"
    )


def test_thrust_roughly_preserved_across_wind_speeds():
    # Documented simplification: CT (and therefore thrust at fixed RPM)
    # is treated as approximately constant across this mu range -- this
    # test just confirms that documented assumption is actually what the
    # code does (thrust shouldn't swing wildly with wind_speed at fixed RPM).
    diameter_m = 10 * 0.0254
    rpm = 5000
    still_air = bemt_thrust_and_power_forward_flight(diameter_m, rpm, SEA_LEVEL_DENSITY, 0)
    windy = bemt_thrust_and_power_forward_flight(diameter_m, rpm, SEA_LEVEL_DENSITY, 10)
    assert still_air["thrust_n"] == windy["thrust_n"], "Expected thrust to be treated as wind-independent by design"
    print(f"PASS: thrust held at {still_air['thrust_n']:.2f}N across wind speeds (documented simplification)")


def test_moderate_wind_shows_translational_lift_benefit():
    # Moderate 8 m/s (~18mph) wind, realistic build -- per the bucket
    # curve above, this lands on the DECREASING side: real rotorcraft
    # aerodynamics predicts less power (and thus LONGER estimated flight
    # time) than dead-still air, not more. Counterintuitive, but this is
    # the documented translational-lift effect, not a bug -- see the
    # module docstring in bemt.py.
    result = analyze_bemt_hover_in_wind(
        mass_kg=1.2, motor_count=4, propeller_diameter_in=10,
        motor_kv=920, battery_cells=4, battery_capacity_mah=5000,
        air_density=SEA_LEVEL_DENSITY, wind_speed_mps=8,
    )
    assert result["power_increase_percent"] < 0, f"Expected a power REDUCTION at moderate wind, got {result}"
    assert (
        result["estimated_flight_time_in_wind_minutes"]
        > result["estimated_flight_time_still_air_minutes"]
    ), "Expected longer flight time at moderate wind (translational lift)"
    print(f"PASS: moderate 8 m/s wind scenario shows translational-lift power reduction -> {result}")


def test_strong_wind_recovers_from_the_bucket_minimum():
    # For this specific build (low disc loading, typical of a small
    # multirotor), the bucket curve's minimum sits around 20-22 m/s and
    # power at 35 m/s -- past the minimum, where profile drag has started
    # to dominate again -- should be MEASURABLY HIGHER than at the
    # minimum, i.e. climbing back up, even though (see the module
    # docstring in bemt.py) it doesn't fully climb back above the still-
    # air hover value within this app's validated 0-40 m/s range for this
    # particular rotor -- translational lift's benefit is simply large
    # enough, for this low a disc loading, that even strong wind stays a
    # net win versus hovering in dead-still air. That's a real,
    # legitimate result of the physics, not a bug -- see bemt.py.
    at_minimum = analyze_bemt_hover_in_wind(
        mass_kg=1.2, motor_count=4, propeller_diameter_in=10,
        motor_kv=920, battery_cells=4, battery_capacity_mah=5000,
        air_density=SEA_LEVEL_DENSITY, wind_speed_mps=22,
    )
    at_strong_wind = analyze_bemt_hover_in_wind(
        mass_kg=1.2, motor_count=4, propeller_diameter_in=10,
        motor_kv=920, battery_cells=4, battery_capacity_mah=5000,
        air_density=SEA_LEVEL_DENSITY, wind_speed_mps=35,
    )
    assert at_strong_wind["power_per_motor_in_wind_w"] > at_minimum["power_per_motor_in_wind_w"], (
        f"Expected power to climb back up past the bucket minimum, "
        f"got {at_minimum['power_per_motor_in_wind_w']}W at 22m/s vs "
        f"{at_strong_wind['power_per_motor_in_wind_w']}W at 35m/s"
    )
    print(
        f"PASS: power climbs from {at_minimum['power_per_motor_in_wind_w']}W (near the bucket minimum, ~22m/s) "
        f"to {at_strong_wind['power_per_motor_in_wind_w']}W at 35m/s -- recovering from the minimum as expected, "
        f"even though it stays below the still-air hover value ({at_strong_wind['power_per_motor_still_air_w']}W) "
        f"across this app's whole validated 0-40 m/s range for this low-disc-loading example"
    )


def test_no_wind_scenario_matches_still_air_analysis():
    # wind_speed_mps=0 through the full analyze_bemt_hover_in_wind
    # pipeline should show a 0% power penalty and identical still-air/
    # in-wind flight times -- another end-to-end confirmation that the
    # zero-wind case is a true no-op, not an approximation that happens
    # to be close.
    result = analyze_bemt_hover_in_wind(
        mass_kg=1.2, motor_count=4, propeller_diameter_in=10,
        motor_kv=920, battery_cells=4, battery_capacity_mah=5000,
        air_density=SEA_LEVEL_DENSITY, wind_speed_mps=0,
    )
    assert result["power_increase_percent"] == 0.0
    assert result["estimated_flight_time_in_wind_minutes"] == result["estimated_flight_time_still_air_minutes"]
    print(f"PASS: zero-wind scenario shows exactly 0% power penalty -> {result}")


if __name__ == "__main__":
    test_zero_wind_reduces_exactly_to_hover()
    test_glauert_inflow_matches_hover_at_zero_advance_ratio()
    test_power_follows_the_classic_bucket_curve_shape()
    test_thrust_roughly_preserved_across_wind_speeds()
    test_moderate_wind_shows_translational_lift_benefit()
    test_strong_wind_recovers_from_the_bucket_minimum()
    test_no_wind_scenario_matches_still_air_analysis()
    print("\nAll BEMT wind/forward-flight tests passed.")
