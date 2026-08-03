"""
Standalone test script for the efficiency landscape heatmap -- run
directly with `python3 test_efficiency_landscape.py`, matching the other
standalone test_*.py scripts in this repo.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.efficiency_landscape import compute_efficiency_landscape
from app.services.parameter_sweep import analyze_sweep_point

REALISTIC_DRONE = dict(
    mass_kg=1.2, motor_count=4, propeller_diameter_in=10,
    motor_kv=920, battery_cells=4, battery_capacity_mah=5000,
)


def test_grid_shape_matches_requested_size():
    result = compute_efficiency_landscape(**REALISTIC_DRONE, grid_size=8)
    assert len(result["diameter_values"]) == 8
    assert len(result["kv_values"]) == 8
    assert len(result["efficiency_grid"]) == 8
    assert all(len(row) == 8 for row in result["efficiency_grid"])
    print(f"PASS: 8x8 grid requested, got {len(result['efficiency_grid'])}x{len(result['efficiency_grid'][0])}")


def test_grid_is_centered_on_the_drones_current_spec():
    result = compute_efficiency_landscape(**REALISTIC_DRONE, grid_size=11, diameter_span_in=4, kv_span=400)
    # With an odd grid size and a symmetric span, the middle index should
    # land exactly on the drone's current spec.
    middle = 5
    assert abs(result["diameter_values"][middle] - 10.0) < 0.01, result["diameter_values"]
    assert abs(result["kv_values"][middle] - 920.0) < 0.01, result["kv_values"]
    print(f"PASS: grid centered correctly -- middle diameter={result['diameter_values'][middle]}, middle KV={result['kv_values'][middle]}")


def test_center_cell_matches_direct_analyze_sweep_point_call():
    # The real correctness check: the grid cell at the drone's own current
    # spec should reproduce parameter_sweep.py's own analysis exactly --
    # confirms no unit-conversion or off-by-one bug snuck into the grid
    # construction, not just "the grid has plausible-looking numbers."
    result = compute_efficiency_landscape(**REALISTIC_DRONE, grid_size=11, diameter_span_in=4, kv_span=400)
    middle = 5
    grid_efficiency = result["efficiency_grid"][middle][middle]

    direct = analyze_sweep_point(**REALISTIC_DRONE)
    assert abs(grid_efficiency - direct["hover_efficiency_w_per_kg"]) < 0.01, (
        f"Grid center efficiency {grid_efficiency} doesn't match direct analyze_sweep_point "
        f"result {direct['hover_efficiency_w_per_kg']}"
    )
    print(f"PASS: grid center efficiency ({grid_efficiency} W/kg) matches direct analyze_sweep_point call")


def test_bigger_propeller_is_more_efficient_in_this_region():
    # Physically expected direction (same one verified in
    # test_parameter_sweep.py): for a fixed KV, a bigger propeller needs
    # less RPM for the same thrust, and reduces required RPM more than
    # profile-drag-side losses grow at these diameters -- so moving along
    # the diameter axis (fixed KV column) should show DECREASING required
    # power, i.e. DECREASING (better) W/kg efficiency.
    result = compute_efficiency_landscape(**REALISTIC_DRONE, grid_size=10)
    kv_column = 5  # arbitrary fixed KV column
    efficiencies_along_diameter = [row[kv_column] for row in result["efficiency_grid"]]
    for i in range(1, len(efficiencies_along_diameter)):
        assert efficiencies_along_diameter[i] < efficiencies_along_diameter[i - 1], (
            f"Expected efficiency to improve (decrease) with increasing propeller diameter, "
            f"got {efficiencies_along_diameter}"
        )
    print(f"PASS: efficiency improves monotonically with propeller diameter along a fixed-KV column -> {[round(e,1) for e in efficiencies_along_diameter]}")


def test_feasible_grid_flags_underpowered_corner():
    # A build with a small prop AND low KV (low end of both axes) should
    # be flagged infeasible somewhere in that corner for a heavy-ish
    # reference drone with a tight span -- confirms feasible_grid carries
    # real information, not just always True.
    result = compute_efficiency_landscape(
        mass_kg=3.0, motor_count=4, propeller_diameter_in=8, motor_kv=400,
        battery_cells=3, battery_capacity_mah=5000, grid_size=8, diameter_span_in=3, kv_span=300,
    )
    assert any(not cell for row in result["feasible_grid"] for cell in row), (
        "Expected at least one infeasible cell in this deliberately underpowered-corner scenario"
    )
    print("PASS: feasible_grid correctly flags at least one infeasible cell in the underpowered corner")


if __name__ == "__main__":
    test_grid_shape_matches_requested_size()
    test_grid_is_centered_on_the_drones_current_spec()
    test_center_cell_matches_direct_analyze_sweep_point_call()
    test_bigger_propeller_is_more_efficient_in_this_region()
    test_feasible_grid_flags_underpowered_corner()
    print("\nAll efficiency landscape tests passed.")
