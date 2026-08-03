from pydantic import BaseModel


class ParameterSweepRequest(BaseModel):
    # Deltas applied to the drone's currently stored spec -- 0 means "no
    # change to this parameter." Matches app/services/parameter_sweep.py's
    # analyze_parameter_sweep() parameter names.
    mass_delta_kg: float = 0
    propeller_diameter_delta_in: float = 0
    motor_kv_delta: float = 0
    battery_cells_delta: int = 0
