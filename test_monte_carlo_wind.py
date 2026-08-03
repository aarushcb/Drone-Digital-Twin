"""
Standalone test script for wind-speed (gust) Monte Carlo uncertainty
propagation -- run directly with `python3 test_monte_carlo_wind.py`,
matching the other standalone test_*.py scripts in this repo.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.monte_carlo_uq import monte_carlo_wind_endurance_uncertainty
from app.services.bemt import analyze_bemt_hover_in_wind

SEA_LEVEL_DENSITY = 1.225

REALISTIC_DRONE = dict(
    mass_kg=1.2, motor_count=4, propeller_diameter_in=10,
    battery_cells=4, battery_capacity_mah=5000, air_density=SEA_LEVEL_DENSITY,
)


def test_same_seed_is_reproducible():
    r1 = monte_carlo_wind_endurance_uncertainty(**REALISTIC_DRONE, mean_wind_speed_mps=8, num_samples=500, seed=1)
    r2 = monte_carlo_wind_endurance_uncertainty(**REALISTIC_DRONE, mean_wind_speed_mps=8, num_samples=500, seed=1)
    assert r1 == r2, "Same seed should produce identical results"
    print(f"PASS: same seed reproducible -> total_power_w_mean={r1['total_power_w_mean']}")


def test_deterministic_point_estimate_falls_inside_the_confidence_interval():
    # analyze_bemt_hover_in_wind() at the SAME mean wind speed (still-air
    # required RPM, forward-flight power at that RPM and wind speed) is
    # this Monte Carlo function's noise-free center case -- it should
    # land inside the reported 90% CI, confirming the two are describing
    # the same underlying model consistently.
    mean_wind = 8.0
    deterministic = analyze_bemt_hover_in_wind(
        mass_kg=1.2, motor_count=4, propeller_diameter_in=10,
        motor_kv=920, battery_cells=4, battery_capacity_mah=5000,
        air_density=SEA_LEVEL_DENSITY, wind_speed_mps=mean_wind,
    )
    deterministic_total_power = deterministic["power_per_motor_in_wind_w"] * 4

    result = monte_carlo_wind_endurance_uncertainty(
        **REALISTIC_DRONE, mean_wind_speed_mps=mean_wind, num_samples=3000, seed=7,
    )
    assert result["total_power_w_p05"] < deterministic_total_power < result["total_power_w_p95"], (
        f"Deterministic power {deterministic_total_power}W falls outside the reported 90% CI "
        f"[{result['total_power_w_p05']}, {result['total_power_w_p95']}]"
    )
    print(
        f"PASS: deterministic total power={deterministic_total_power:.1f}W inside 90% CI "
        f"[{result['total_power_w_p05']}, {result['total_power_w_p95']}]"
    )


def test_convergence_with_more_samples():
    small = monte_carlo_wind_endurance_uncertainty(**REALISTIC_DRONE, mean_wind_speed_mps=10, num_samples=2000, seed=1)
    large = monte_carlo_wind_endurance_uncertainty(**REALISTIC_DRONE, mean_wind_speed_mps=10, num_samples=5000, seed=2)
    relative_diff = abs(small["total_power_w_mean"] - large["total_power_w_mean"]) / large["total_power_w_mean"]
    assert relative_diff < 0.02, f"Mean power changed {relative_diff:.2%} going from 2000 to 5000 samples"
    print(
        f"PASS: converged -- power mean at 2000 samples={small['total_power_w_mean']}W, "
        f"at 5000 samples={large['total_power_w_mean']}W (relative diff={relative_diff:.3%})"
    )


def test_higher_turbulence_widens_the_confidence_interval():
    # More gust variability around the same mean wind speed should widen
    # (not narrow or leave unchanged) the resulting power/endurance
    # uncertainty band -- the basic sanity check that this is actually
    # propagating variance, not just adding noise that washes out.
    calm = monte_carlo_wind_endurance_uncertainty(
        **REALISTIC_DRONE, mean_wind_speed_mps=10, turbulence_intensity=0.05, num_samples=3000, seed=3,
    )
    gusty = monte_carlo_wind_endurance_uncertainty(
        **REALISTIC_DRONE, mean_wind_speed_mps=10, turbulence_intensity=0.35, num_samples=3000, seed=3,
    )
    calm_width = calm["total_power_w_p95"] - calm["total_power_w_p05"]
    gusty_width = gusty["total_power_w_p95"] - gusty["total_power_w_p05"]
    assert gusty_width > calm_width, (
        f"Expected higher turbulence intensity to widen the CI, got calm width={calm_width}, gusty width={gusty_width}"
    )
    print(f"PASS: 90% CI width grows with turbulence intensity -- calm(5%)={calm_width:.1f}W, gusty(35%)={gusty_width:.1f}W")


def test_zero_wind_collapses_to_a_tight_distribution_at_hover_power():
    # At mean_wind_speed=0, gust_std is also 0 (turbulence scales with
    # mean wind speed in this model) -- every sample should draw exactly
    # 0 m/s wind, so the distribution should collapse to a single point
    # matching the still-air hover power, with zero spread.
    result = monte_carlo_wind_endurance_uncertainty(
        **REALISTIC_DRONE, mean_wind_speed_mps=0, num_samples=500, seed=1,
    )
    assert result["total_power_w_std_dev"] == 0.0, f"Expected zero spread at zero mean wind, got {result}"
    print(f"PASS: zero mean wind speed collapses to a single deterministic power value ({result['total_power_w_mean']}W)")


if __name__ == "__main__":
    test_same_seed_is_reproducible()
    test_deterministic_point_estimate_falls_inside_the_confidence_interval()
    test_convergence_with_more_samples()
    test_higher_turbulence_widens_the_confidence_interval()
    test_zero_wind_collapses_to_a_tight_distribution_at_hover_power()
    print("\nAll Monte Carlo wind uncertainty tests passed.")
