"""
Standalone test script for Monte Carlo uncertainty quantification --
run directly with `python3 test_monte_carlo_uq.py`, matching the other
standalone test_*.py scripts in this repo.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.monte_carlo_uq import monte_carlo_ct_uncertainty, monte_carlo_hover_uncertainty
from app.services.bemt import hover_coefficients, bemt_required_rpm_for_thrust

SEA_LEVEL_DENSITY = 1.225


def test_same_seed_is_reproducible():
    # A Monte Carlo result MUST be reproducible for a fixed seed, or it's
    # not trustworthy/debuggable -- this is a basic correctness property,
    # not just a plausibility check.
    r1 = monte_carlo_ct_uncertainty(num_samples=500, seed=42)
    r2 = monte_carlo_ct_uncertainty(num_samples=500, seed=42)
    assert r1 == r2, "Same seed should produce identical results"
    print(f"PASS: same seed reproducible -> ct_mean={r1['ct_mean']:.5f}")


def test_nominal_point_estimate_falls_inside_the_confidence_interval():
    # The existing single-point BEMT value (already verified in
    # test_bemt.py) should land inside the 90% CI this module reports --
    # if it didn't, that would mean the "representative" midpoint values
    # in bemt.py aren't actually representative of their own documented
    # ranges, a real inconsistency bug.
    nominal_ct = hover_coefficients()["ct"]
    result = monte_carlo_ct_uncertainty(num_samples=3000, seed=7)
    assert result["ct_p05"] < nominal_ct < result["ct_p95"], (
        f"Nominal CT {nominal_ct} falls outside the reported 90% CI "
        f"[{result['ct_p05']}, {result['ct_p95']}]"
    )
    print(
        f"PASS: nominal CT={nominal_ct:.5f} inside 90% CI "
        f"[{result['ct_p05']:.5f}, {result['ct_p95']:.5f}]"
    )


def test_convergence_with_more_samples():
    # Confirms 2000 samples (the default used by the API endpoint) has
    # actually converged -- the reported mean shouldn't meaningfully
    # change with 2.5x more samples. A Monte Carlo result that's still
    # drifting with sample count would be an honesty problem (the
    # "confidence interval" would itself be unreliable).
    small = monte_carlo_ct_uncertainty(num_samples=2000, seed=1)
    large = monte_carlo_ct_uncertainty(num_samples=5000, seed=2)
    relative_diff = abs(small["ct_mean"] - large["ct_mean"]) / large["ct_mean"]
    assert relative_diff < 0.01, f"Mean CT changed {relative_diff:.2%} going from 2000 to 5000 samples"
    print(
        f"PASS: converged -- ct_mean at 2000 samples={small['ct_mean']:.5f}, "
        f"at 5000 samples={large['ct_mean']:.5f} (relative diff={relative_diff:.3%})"
    )


def test_hover_rpm_uncertainty_brackets_the_point_estimate():
    # Same cross-check as above, but for the dimensional, drone-specific
    # quantity (required RPM) -- using the same realistic reference build
    # as test_bemt.py.
    diameter_m = 10 * 0.0254
    thrust_per_motor = (1.2 * 9.81) / 4
    nominal_rpm = bemt_required_rpm_for_thrust(thrust_per_motor, SEA_LEVEL_DENSITY, diameter_m)

    result = monte_carlo_hover_uncertainty(
        mass_kg=1.2, motor_count=4, propeller_diameter_in=10,
        air_density=SEA_LEVEL_DENSITY, num_samples=3000, seed=99,
    )
    assert result["required_rpm_p05"] < nominal_rpm < result["required_rpm_p95"], (
        f"Nominal required RPM {nominal_rpm} falls outside the reported 90% CI"
    )
    print(
        f"PASS: nominal required_rpm={nominal_rpm:.0f} inside 90% CI "
        f"[{result['required_rpm_p05']}, {result['required_rpm_p95']}] "
        f"(mean={result['required_rpm_mean']}, std_dev={result['required_rpm_std_dev']})"
    )


if __name__ == "__main__":
    test_same_seed_is_reproducible()
    test_nominal_point_estimate_falls_inside_the_confidence_interval()
    test_convergence_with_more_samples()
    test_hover_rpm_uncertainty_brackets_the_point_estimate()
    print("\nAll Monte Carlo UQ tests passed.")
