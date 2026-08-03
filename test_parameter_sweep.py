"""
Standalone test script for the interactive parameter sweep -- run
directly with `python3 test_parameter_sweep.py`, matching the other
standalone test_*.py scripts in this repo.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.parameter_sweep import analyze_sweep_point, analyze_parameter_sweep
from app.services.motor_performance import required_rpm_for_thrust

REALISTIC_DRONE = dict(
    mass_kg=1.2, motor_count=4, propeller_diameter_in=10,
    motor_kv=920, battery_cells=4, battery_capacity_mah=5000,
)


def test_matches_hand_verified_reference_scenario():
    # Same realistic scenario already hand-verified in
    # test_motor_performance.py (71.6W total hover power) -- confirms
    # this module's reuse of the underlying formulas didn't drift.
    result = analyze_sweep_point(**REALISTIC_DRONE, air_density=1.225)
    assert result["total_hover_power_watts"] == 71.6, f"Expected 71.6W (matches test_motor_performance.py), got {result['total_hover_power_watts']}"
    expected_efficiency = 71.6 / 1.2
    assert abs(result["hover_efficiency_w_per_kg"] - expected_efficiency) < 0.05, (
        f"Expected ~{expected_efficiency:.2f} W/kg, got {result['hover_efficiency_w_per_kg']}"
    )
    print(f"PASS: hover efficiency = {result['hover_efficiency_w_per_kg']} W/kg (matches 71.6W / 1.2kg reference)")


def test_max_altitude_bisection_converges_to_the_correct_crossover():
    # The real correctness check: at the RETURNED max_altitude_m, required
    # RPM should land within ~1 RPM of max_available_rpm -- confirms the
    # bisection actually converged to the crossover point, not just that
    # it ran without crashing. Uses a lower cell count than the main
    # reference scenario specifically so there IS an interior crossover
    # within the search range (the main reference scenario has such a
    # large RPM margin it never runs out of altitude within 9000m --
    # exercised separately below).
    tight_margin_build = dict(REALISTIC_DRONE, battery_cells=2)
    result = analyze_sweep_point(**tight_margin_build, air_density=1.225)
    assert result["altitude_search_capped"] is False, f"Expected an interior crossover altitude, got {result}"

    from app.services.parameter_sweep import _standard_temp_at_altitude_c
    from app.services.environment_simulator import air_density as isa_air_density

    diameter_m = 10 * 0.0254
    thrust_per_motor = (1.2 * 9.81) / 4
    h = result["max_altitude_m"]
    density_at_h = isa_air_density(h, _standard_temp_at_altitude_c(h))
    rpm_at_h = required_rpm_for_thrust(thrust_per_motor, density_at_h, diameter_m)

    assert abs(rpm_at_h - result["max_available_rpm"]) < 1.0, (
        f"Expected required RPM at max_altitude_m to match max_available_rpm within 1 RPM, "
        f"got required={rpm_at_h:.2f}, max_available={result['max_available_rpm']}"
    )
    print(f"PASS: max_altitude_m={h}m converges correctly (required RPM there = {rpm_at_h:.2f}, max available = {result['max_available_rpm']})")


def test_infeasible_at_sea_level_returns_zero_altitude():
    # Deliberately underpowered build (same as test_motor_performance.py's
    # infeasibility test) -- can't even hover at sea level, so max
    # altitude should honestly be reported as 0, not a fabricated number.
    result = analyze_sweep_point(
        mass_kg=5.0, motor_count=4, propeller_diameter_in=10,
        motor_kv=200, battery_cells=2, battery_capacity_mah=5000, air_density=1.225,
    )
    assert result["feasible"] is False
    assert result["max_altitude_m"] == 0.0
    assert result["altitude_search_capped"] is False
    print(f"PASS: underpowered build correctly reports max_altitude_m=0 (infeasible even at sea level)")


def test_overpowered_build_hits_the_search_ceiling_honestly():
    # A build with a huge RPM margin should still be feasible even at the
    # search ceiling -- the function should say so honestly (capped, not
    # a fabricated precise number) rather than claiming a specific ceiling
    # it never actually found.
    result = analyze_sweep_point(
        mass_kg=0.5, motor_count=4, propeller_diameter_in=10,
        motor_kv=920, battery_cells=6, battery_capacity_mah=5000, air_density=1.225,
    )
    assert result["altitude_search_capped"] is True
    assert result["max_altitude_m"] == 9000.0
    print(f"PASS: overpowered build honestly reports search-capped at {result['max_altitude_m']}m rather than a fabricated ceiling")


def test_current_vs_swept_comparison_shows_expected_direction():
    # Bumping propeller diameter up (more thrust per RPM) with everything
    # else fixed should make hovering EASIER: lower required RPM, lower
    # hover power, better (lower) W/kg efficiency, longer flight time,
    # higher max altitude. A real, specific, testable direction -- not
    # just "some numbers changed."
    result = analyze_parameter_sweep(**REALISTIC_DRONE, propeller_diameter_delta_in=2.0)
    current, swept = result["current"], result["swept"]

    assert swept["required_rpm"] < current["required_rpm"]
    assert swept["hover_efficiency_w_per_kg"] < current["hover_efficiency_w_per_kg"]
    assert swept["estimated_flight_time_minutes"] > current["estimated_flight_time_minutes"]
    assert swept["max_altitude_m"] >= current["max_altitude_m"]
    assert result["swept_spec"]["propeller_diameter_in"] == 12.0
    print(
        f"PASS: +2in propeller -> required_rpm {current['required_rpm']}->{swept['required_rpm']}, "
        f"efficiency {current['hover_efficiency_w_per_kg']}->{swept['hover_efficiency_w_per_kg']} W/kg, "
        f"flight time {current['estimated_flight_time_minutes']}->{swept['estimated_flight_time_minutes']}min"
    )


def test_adding_mass_makes_things_worse():
    # The opposite-direction sanity check: strapping on more mass (e.g. a
    # gimbal/payload) with nothing else changed should make efficiency
    # WORSE (higher W/kg) and flight time shorter -- the physically
    # correct direction.
    result = analyze_parameter_sweep(**REALISTIC_DRONE, mass_delta_kg=0.5)
    current, swept = result["current"], result["swept"]
    assert swept["hover_efficiency_w_per_kg"] > current["hover_efficiency_w_per_kg"]
    assert swept["estimated_flight_time_minutes"] < current["estimated_flight_time_minutes"]
    print(
        f"PASS: +0.5kg payload -> efficiency worsens {current['hover_efficiency_w_per_kg']}->{swept['hover_efficiency_w_per_kg']} W/kg, "
        f"flight time drops {current['estimated_flight_time_minutes']}->{swept['estimated_flight_time_minutes']}min"
    )


if __name__ == "__main__":
    test_matches_hand_verified_reference_scenario()
    test_max_altitude_bisection_converges_to_the_correct_crossover()
    test_infeasible_at_sea_level_returns_zero_altitude()
    test_overpowered_build_hits_the_search_ceiling_honestly()
    test_current_vs_swept_comparison_shows_expected_direction()
    test_adding_mass_makes_things_worse()
    print("\nAll parameter sweep tests passed.")
