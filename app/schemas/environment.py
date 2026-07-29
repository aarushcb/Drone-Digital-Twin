from pydantic import BaseModel, field_validator


class EnvironmentSimulationRequest(BaseModel):
    altitude_m: float = 0
    temperature_c: float = 15

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
