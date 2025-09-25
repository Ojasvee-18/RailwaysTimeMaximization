"""
Train dynamics and profiles.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timedelta


@dataclass
class TrainProfile:
    acceleration_ms2: float = 0.3
    deceleration_ms2: float = 0.4
    max_speed_kmph: float = 100.0


@dataclass
class Train:
    number: str
    name: str
    route_stations: List[str]
    departure_time: datetime
    dwell_minutes: float = 2.0
    profile: TrainProfile = field(default_factory=TrainProfile)

    def eta_after_distance(self, distance_km: float) -> datetime:
        # Simple kinematic approx: cruise at max speed
        speed_kmph = max(10.0, self.profile.max_speed_kmph)
        minutes = (distance_km / speed_kmph) * 60.0
        return self.departure_time + timedelta(minutes=minutes)


