from pydantic import BaseModel
from typing import Optional, List


class CalibrationCheckRequest(BaseModel):
    # Each field is one calibration step's logged readings -- all
    # optional, since the step-by-step Flutter UI calls this once per
    # completed step, not necessarily all four at once. See
    # app/services/sensor_calibration.py for what each expects physically
    # (accel/gyro: stationary readings; compass: readings across a slow
    # rotation through many headings; esc: one reading per motor at the
    # same commanded throttle).
    accel_x: Optional[List[float]] = None
    accel_y: Optional[List[float]] = None
    accel_z: Optional[List[float]] = None
    gyro_x: Optional[List[float]] = None
    gyro_y: Optional[List[float]] = None
    gyro_z: Optional[List[float]] = None
    mag_x: Optional[List[float]] = None
    mag_y: Optional[List[float]] = None
    mag_z: Optional[List[float]] = None
    esc_readings: Optional[List[float]] = None
