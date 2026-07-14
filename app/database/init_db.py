from app.database.database import engine

from app.models.drone import Base
from app.models.telemetry import Telemetry

Base.metadata.create_all(bind=engine)

print("Tables created successfully")