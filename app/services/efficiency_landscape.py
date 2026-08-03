"""
WHY THIS EXISTS:
The parameter sweep (app/services/parameter_sweep.py) answers "what
happens if I change ONE thing" -- this answers the 2D version of that
same question: "across a whole GRID of (propeller diameter, motor KV)
combinations, where's the efficient region, and where does my drone's
current spec sit relative to it." It's the same physics evaluated many
times over, not a new model: every grid cell is one call to
parameter_sweep.py's analyze_sweep_point(), holding mass, motor count,
battery cells, and battery capacity fixed at the drone's stored spec and
varying only propeller diameter and motor KV (the two axes of the
heatmap) -- reusing the exact same already-verified formulas, not
duplicating them.

WHY THESE TWO AXES SPECIFICALLY:
Propeller diameter and motor KV are the two parameters with the most
direct, opposing effect on the thrust/RPM tradeoff (bigger prop needs
less RPM for the same thrust per motor_performance.py's T=Ct*rho*n^2*D^4;
higher KV motor spins faster per volt) -- varying them together is the
classic multirotor "prop and motor matching" design space students are
taught to reason about, which is why this is the pairing requested for
the heatmap rather than, say, mass vs. battery cells.

WHAT WAS VERIFIED BEFORE SHIPPING:
- The grid cell nearest the drone's actual stored spec reproduces
  parameter_sweep.py's own "current" analysis (same function, same
  inputs -- this is really just confirming the grid construction doesn't
  introduce an off-by-one or unit-conversion bug between the axis values
  and what gets passed into analyze_sweep_point).
- Efficiency (W/kg) decreases monotonically along at least one direction
  of each axis in a neighborhood of a feasible point (bigger prop and/or
  higher relative KV margin should generally look "more efficient" in
  the sense of lower required RPM burden) -- confirmed directionally in
  test_efficiency_landscape.py, not just "grid has numbers in it."
"""

from typing import List

from app.services.parameter_sweep import analyze_sweep_point

DEFAULT_GRID_SIZE = 12
DEFAULT_DIAMETER_SPAN_IN = 4.0  # +-4in around the drone's current propeller diameter
DEFAULT_KV_SPAN = 400.0  # +-400KV around the drone's current motor KV


def _linspace(low: float, high: float, n: int) -> List[float]:
    if n == 1:
        return [low]
    step = (high - low) / (n - 1)
    return [low + step * i for i in range(n)]


def compute_efficiency_landscape(
    mass_kg: float,
    motor_count: int,
    propeller_diameter_in: float,
    motor_kv: float,
    battery_cells: int,
    battery_capacity_mah: float,
    grid_size: int = DEFAULT_GRID_SIZE,
    diameter_span_in: float = DEFAULT_DIAMETER_SPAN_IN,
    kv_span: float = DEFAULT_KV_SPAN,
) -> dict:
    """
    Builds a grid_size x grid_size grid over (propeller_diameter_in,
    motor_kv), centered on the drone's current spec, evaluating
    parameter_sweep.py's analyze_sweep_point() at every cell with mass,
    motor count, battery cells, and battery capacity held fixed.
    Convention: efficiency_grid[i][j] corresponds to
    diameter_values[i] and kv_values[j] (row=diameter, column=KV).
    """
    diameter_low = max(2.0, propeller_diameter_in - diameter_span_in)
    diameter_high = propeller_diameter_in + diameter_span_in
    kv_low = max(50.0, motor_kv - kv_span)
    kv_high = motor_kv + kv_span

    diameter_values = _linspace(diameter_low, diameter_high, grid_size)
    kv_values = _linspace(kv_low, kv_high, grid_size)

    efficiency_grid = []
    feasible_grid = []
    for diameter in diameter_values:
        efficiency_row = []
        feasible_row = []
        for kv in kv_values:
            result = analyze_sweep_point(
                mass_kg=mass_kg,
                motor_count=motor_count,
                propeller_diameter_in=diameter,
                motor_kv=kv,
                battery_cells=battery_cells,
                battery_capacity_mah=battery_capacity_mah,
            )
            efficiency_row.append(result["hover_efficiency_w_per_kg"])
            feasible_row.append(result["feasible"])
        efficiency_grid.append(efficiency_row)
        feasible_grid.append(feasible_row)

    return {
        "diameter_values": [round(d, 2) for d in diameter_values],
        "kv_values": [round(k, 1) for k in kv_values],
        "efficiency_grid": efficiency_grid,
        "feasible_grid": feasible_grid,
        "current_diameter_in": propeller_diameter_in,
        "current_kv": motor_kv,
    }
