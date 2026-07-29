import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.motor_performance import (
    analyze_motor_performance,
    required_rpm_for_thrust,
    ideal_hover_power_watts,
)

SEA_LEVEL_DENSITY = 1.225


def test_required_rpm_matches_hand_verified_reference():
    # Same scenario manually verified before writing this file: 1.2kg
    # quad, 10" props, sea-level air -> should land in the real-world
    # plausible range for small multirotor hover RPM.
    thrust_per_motor = (1.2 * 9.81) / 4
    diameter_m = 10 * 0.0254
    rpm = required_rpm_for_thrust(thrust_per_motor, SEA_LEVEL_DENSITY, diameter_m)
    assert 3500 < rpm < 5500, f"Expected a realistic hover RPM range, got {rpm}"
    print(f"PASS: required RPM = {rpm:.0f} (within realistic 3500-5500 range for this config)")


def test_realistic_drone_is_feasible():
    # A realistic, well-matched small quad build -- should be able to hover.
    result = analyze_motor_performance(
        mass_kg=1.2,
        motor_count=4,
        propeller_diameter_in=10,
        motor_kv=920,       # common small-quad motor KV
        battery_cells=4,    # 4S, 14.8V nominal
        battery_capacity_mah=5000,
        air_density=SEA_LEVEL_DENSITY,
    )
    assert result["feasible"] is True, f"Expected a well-matched build to be feasible, got {result}"
    assert result["estimated_flight_time_minutes"] is not None
    assert result["estimated_flight_time_minutes"] > 0
    print(f"PASS: realistic build is feasible -> {result}")


def test_underpowered_motor_is_correctly_flagged_infeasible():
    # Deliberately mismatched: a heavy drone with a low-KV motor on a
    # low-cell-count (low voltage) battery -- shouldn't be able to reach
    # the RPM needed to hover. This proves the feasibility check actually
    # DOES something, rather than always returning True.
    result = analyze_motor_performance(
        mass_kg=5.0,           # heavy
        motor_count=4,
        propeller_diameter_in=10,
        motor_kv=200,           # low KV -- built for torque, not high RPM
        battery_cells=2,        # only 2S -- low voltage, low max RPM
        battery_capacity_mah=5000,
        air_density=SEA_LEVEL_DENSITY,
    )
    assert result["feasible"] is False, f"Expected an underpowered build to be flagged infeasible, got {result}"
    print(f"PASS: underpowered build correctly flagged infeasible -> {result}")


def test_thinner_air_requires_more_power_and_rpm():
    # Same drone, but at a high-altitude/thin-air density -- should need
    # MORE RPM and MORE power than at sea level (physically correct
    # direction of effect).
    sea_level_result = analyze_motor_performance(
        mass_kg=1.2, motor_count=4, propeller_diameter_in=10, motor_kv=920,
        battery_cells=4, battery_capacity_mah=5000, air_density=SEA_LEVEL_DENSITY,
    )
    thin_air_density = 0.95  # roughly 3000m altitude
    thin_air_result = analyze_motor_performance(
        mass_kg=1.2, motor_count=4, propeller_diameter_in=10, motor_kv=920,
        battery_cells=4, battery_capacity_mah=5000, air_density=thin_air_density,
    )
    assert thin_air_result["required_rpm"] > sea_level_result["required_rpm"]
    assert thin_air_result["total_hover_power_watts"] > sea_level_result["total_hover_power_watts"]
    assert thin_air_result["estimated_flight_time_minutes"] < sea_level_result["estimated_flight_time_minutes"]
    print(
        f"PASS: thin air needs more RPM ({sea_level_result['required_rpm']} -> {thin_air_result['required_rpm']}) "
        f"and more power ({sea_level_result['total_hover_power_watts']}W -> {thin_air_result['total_hover_power_watts']}W), "
        f"shorter flight time ({sea_level_result['estimated_flight_time_minutes']}min -> {thin_air_result['estimated_flight_time_minutes']}min)"
    )


if __name__ == "__main__":
    test_required_rpm_matches_hand_verified_reference()
    test_realistic_drone_is_feasible()
    test_underpowered_motor_is_correctly_flagged_infeasible()
    test_thinner_air_requires_more_power_and_rpm()
    print("\nAll motor performance tests passed.")
