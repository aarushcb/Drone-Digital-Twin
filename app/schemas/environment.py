from pydantic import BaseModel, field_validator


class EnvironmentSimulationRequest(BaseModel):
    altitude_m: float = 0
    temperature_c: float = 15

    # Mean sustained wind speed, in m/s, the drone would need to hold
    # position against -- defaults to 0 (still air), which keeps every
    # existing caller's behavior byte-for-byte unchanged (see
    # app/services/bemt.py's forward-flight functions: at wind_speed=0
    # they collapse exactly to the existing hover-only results).
    wind_speed_mps: float = 0

    # WHY THESE BOUNDS: keeps the barometric formula (valid for the
    # troposphere, roughly up to 11km) from being fed inputs it was
    # never designed for, and keeps temperature within a range real
    # drone flights would plausibly occur in.
    @field_validator("altitude_m")
    @classmethod
    def validate_altitude(cls, v):
        if not (-500 <= v <= 9000):
            raise ValueError("altitude_m must be between -500 and 9000 (the barometric formula used here is only valid within the troposphere)")
        return v

    @field_validator("temperature_c")
    @classmethod
    def validate_temperature(cls, v):
        if not (-60 <= v <= 60):
            raise ValueError("temperature_c must be between -60 and 60 -- outside any realistic flight condition")
        return v

    @field_validator("wind_speed_mps")
    @classmethod
    def validate_wind_speed(cls, v):
        # 40 m/s (~90mph) is well past what any small consumer/hobby
        # multirotor is rated to fly in -- the Glauert forward-flight
        # inflow solver below also isn't validated for advance ratios
        # that extreme.
        if not (0 <= v <= 40):
            raise ValueError("wind_speed_mps must be between 0 and 40")
        return v
