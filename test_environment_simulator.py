import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.environment_simulator import (
    air_density,
    hover_power_scale_factor,
    battery_capacity_factor,
    simulate_conditions,
    SEA_LEVEL_DENSITY,
)


def test_sea_level_density_matches_known_isa_reference():
    # The textbook ISA sea-level density is 1.225 kg/m^3 at 15C -- this
    # is a hard external reference point, not something we chose.
    rho = air_density(0, 15)
    assert abs(rho - 1.225) < 0.001, f"Expected ~1.225, got {rho}"
    print(f"PASS: sea level density = {rho:.4f} kg/m^3 (matches known ISA reference 1.225)")


def test_3000m_density_matches_known_isa_reference():
    # At 3000m under standard atmosphere conditions, known reference
    # density is ~0.909 kg/m^3.
    standard_temp_at_3000m = 15 - 0.0065 * 3000
    rho = air_density(3000, standard_temp_at_3000m)
    assert abs(rho - 0.909) < 0.01, f"Expected ~0.909, got {rho}"
    print(f"PASS: 3000m density = {rho:.4f} kg/m^3 (matches known ISA reference ~0.909)")


def test_thinner_air_requires_more_hover_power():
    # High altitude + hot temperature = thinner air = more power needed
    # to hover -- should be a factor greater than 1.
    scale = hover_power_scale_factor(altitude_m=2500, temperature_c=35)
    assert scale > 1.0, f"Expected more power needed in thin air, got scale={scale}"
    print(f"PASS: hot/high-altitude power scale factor = {scale:.3f}x (more power needed)")


def test_denser_air_requires_less_hover_power():
    # Cold + low altitude = denser air = LESS power needed than the
    # sea-level/15C baseline.
    scale = hover_power_scale_factor(altitude_m=0, temperature_c=-10)
    assert scale < 1.0, f"Expected less power needed in denser air, got scale={scale}"
    print(f"PASS: cold/sea-level power scale factor = {scale:.3f}x (less power needed)")


def test_battery_capacity_factor_matches_cited_reference_points():
    # These are the actual cited reference points from real sources --
    # confirms the interpolation returns exactly what was cited, not an
    # off-by-one error in the piecewise logic.
    assert abs(battery_capacity_factor(25) - 1.0) < 0.001
    assert abs(battery_capacity_factor(-20) - 0.50) < 0.001
    assert abs(battery_capacity_factor(-10) - 0.70) < 0.001
    # Extreme cold beyond the last data point should clamp, not extrapolate wildly negative
    assert battery_capacity_factor(-50) == 0.50
    print("PASS: battery capacity factor matches all cited reference points")


def test_projected_drain_rate_worsens_in_bad_conditions():
    baseline = 5.0  # 5%/min, a realistic drain rate
    result = simulate_conditions(altitude_m=3000, temperature_c=-15, baseline_drain_rate_percent_per_min=baseline)
    assert result["projected_drain_rate_percent_per_min"] > baseline, (
        f"Expected worse drain rate in cold high-altitude conditions, got {result}"
    )
    print(f"PASS: projected drain rate worsens appropriately -> {result}")


def test_no_baseline_gracefully_omits_projection():
    result = simulate_conditions(altitude_m=1000, temperature_c=20, baseline_drain_rate_percent_per_min=None)
    assert result["projected_drain_rate_percent_per_min"] is None
    assert result["air_density_kg_m3"] is not None  # physics factors still reported
    print("PASS: missing baseline drain rate gracefully omits projection, not a crash")


if __name__ == "__main__":
    test_sea_level_density_matches_known_isa_reference()
    test_3000m_density_matches_known_isa_reference()
    test_thinner_air_requires_more_hover_power()
    test_denser_air_requires_less_hover_power()
    test_battery_capacity_factor_matches_cited_reference_points()
    test_projected_drain_rate_worsens_in_bad_conditions()
    test_no_baseline_gracefully_omits_projection()
    print("\nAll environment simulator tests passed.")
