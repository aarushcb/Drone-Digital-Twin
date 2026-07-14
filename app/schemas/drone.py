from pydantic import BaseModel


class DroneBase(BaseModel):
    name: str
    model: str
    status: str


class DroneCreate(DroneBase):
    pass


class DroneResponse(DroneBase):
    id: int

    class Config:
        from_attributes = True