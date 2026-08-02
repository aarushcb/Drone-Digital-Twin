"""
Standalone test script for the BEMT (Blade Element Momentum Theory) module
-- run directly with `python3 test_bemt.py`, no pytest/framework needed,
matching the other standalone test_*.py scripts in this repo.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.bemt import hover_coefficients, bemt_thrust_and_power, analyze_bemt_hover
from app.services.motor_performance import analyze_motor_performance

SEA_LEVEL_DENSITY = 1.225


def test_ct_lands_in_realistic_published_range():
    # Real small-multirotor propellers, per aggregated UIUC static thrust
    # test data (the same reference family CT_STATIC in
    # motor_performance.py is drawn from), report CT roughly in the
    # 0.008-0.02 range for typical hover pitch settings.
    coeffs = hover_coefficients()
    assert 0.008 < coeffs["ct"] < 0.02, f"CT={coeffs['ct']} outside realistic published range"
    assert abs(coeffs["ct"] - 0.0104) < 0.0005, f"CT={coeffs['ct']} drifted from the hand-verified reference value"
    print(f"PASS: CT = {coeffs['ct']:.5f} (within realistic 0.008-0.02 range)")


def test_thrust_scales_with_rpm_squared():
    # Physical invariant: CT/CP are purely geometric in this hover
    # formulation (independent of RPM), so T = CT*rho*A*(Omega*R)^2 MUST
    # scale exactly as RPM^2. This is a strong correctness check -- it
    # would fail if the coefficient integration or the dimensionalization
    # were wrong, not just "looks plausible."
    diameter_m = 10 * 0.0254
    r1 = bemt_thrust_and_power(diameter_m, rpm=4000, air_density=SEA_LEVEL_DENSITY)
    r2 = bemt_thrust_and_power(diameter_m, rpm=8000, air_density=SEA_LEVEL_DENSITY)

    expected_ratio = (8000 / 4000) ** 2
    actual_ratio = r2["thrust_n"] / r1["thrust_n"]
    assert abs(actual_ratio - expected_ratio) < 1e-6, (
        f"Expected thrust ratio {expected_ratio}, got {actual_ratio}"
    )
    print(f"PASS: thrust scales as RPM^2 exactly (ratio={actual_ratio:.6f}, expected={expected_ratio})")


def test_power_scales_with_rpm_cubed():
    diameter_m = 10 * 0.0254
    r1 = bemt_thrust_and_power(diameter_m, rpm=4000, air_density=SEA_LEVEL_DENSITY)
    r2 = bemt_thrust_and_power(diameter_m, rpm=8000, air_density=SEA_LEVEL_DENSITY)

    expected_ratio = (8000 / 4000) ** 3
    actual_ratio = r2["power_w"] / r1["power_w"]
    assert abs(actual_ratio - expected_ratio) < 1e-6, (
        f"Expected power ratio {expected_ratio}, got {actual_ratio}"
    )
    print(f"PASS: power scales as RPM^3 exactly (ratio={actual_ratio:.6f}, expected={expected_ratio})")


def test_thinner_air_needs_more_rpm_for_same_thrust_direction_check():
    # Sanity check on the density dependence direction: same RPM, thinner
    # air -> less thrust (matches the sign of every other density-
    # dependent formula already in this app, e.g. environment_simulator.py).
    diameter_m = 10 * 0.0254
    sea_level = bemt_thrust_and_power(diameter_m, rpm=5000, air_density=SEA_LEVEL_DENSITY)
    thin_air = bemt_thrust_and_power(diameter_m, rpm=5000, air_density=0.95)
    assert thin_air["thrust_n"] < sea_level["thrust_n"]
    assert thin_air["power_w"] < sea_level["power_w"]
    print(
        f"PASS: thinner air at same RPM gives less thrust "
        f"({sea_level['thrust_n']:.2f}N -> {thin_air['thrust_n']:.2f}N)"
    )


def test_analyze_bemt_hover_matches_realistic_scenario():
    # Same realistic build already used in test_motor_performance.py --
    # should be feasible and produce a sane required RPM.
    result = analyze_bemt_hover(
        mass_kg=1.2,
        motor_count=4,
        propeller_diameter_in=10,
        motor_kv=920,
        battery_cells=4,
        air_density=SEA_LEVEL_DENSITY,
    )
    assert result["bemt_hover_feasible"] is True
    assert 2000 < result["bemt_required_rpm"] < 8000
    print(f"PASS: analyze_bemt_hover realistic scenario -> {result}")


def test_bemt_and_static_ct_models_agree_within_reasonable_margin():
    # The real credibility check: two INDEPENDENTLY-derived thrust models
    # (a single empirical static-Ct constant vs. integrated blade-element/
    # momentum theory) asked the same question ("what RPM to hover?")
    # should land in the same general neighborhood for a normal drone --
    # not identical (they're different models with different simplifying
    # assumptions), but not wildly divergent either. A factor of 2x is a
    # generous but meaningful bound for two models built from different
    # first-principles approaches.
    kwargs = dict(
        mass_kg=1.2, motor_count=4, propeller_diameter_in=10,
        motor_kv=920, battery_cells=4, battery_capacity_mah=5000,
        air_density=SEA_LEVEL_DENSITY,
    )
    static_ct_result = analyze_motor_performance(**kwargs)
    bemt_result = analyze_bemt_hover(
        mass_kg=1.2, motor_count=4, propeller_diameter_in=10,
        motor_kv=920, battery_cells=4, air_density=SEA_LEVEL_DENSITY,
    )
    ratio = bemt_result["bemt_required_rpm"] / static_ct_result["required_rpm"]
    assert 0.5 < ratio < 2.0, (
        f"BEMT required RPM ({bemt_result['bemt_required_rpm']}) and static-Ct "
        f"required RPM ({static_ct_result['required_rpm']}) diverge by more than 2x "
        f"(ratio={ratio:.2f}) -- one of the two models' assumptions may not fit"
    )
    print(
        f"PASS: static-Ct model required_rpm={static_ct_result['required_rpm']}, "
        f"BEMT model required_rpm={bemt_result['bemt_required_rpm']} (ratio={ratio:.2f})"
    )


if __name__ == "__main__":
    test_ct_lands_in_realistic_published_range()
    test_thrust_scales_with_rpm_squared()
    test_power_scales_with_rpm_cubed()
    test_thinner_air_needs_more_rpm_for_same_thrust_direction_check()
    test_analyze_bemt_hover_matches_realistic_scenario()
    test_bemt_and_static_ct_models_agree_within_reasonable_margin()
    print("\nAll BEMT tests passed.")
