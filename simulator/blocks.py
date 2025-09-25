"""
Railway topology primitives: stations, track sections, and blocks.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Station:
    code: str
    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    platforms: int = 2


@dataclass
class TrackSection:
    id: str
    from_station: str
    to_station: str
    length_km: float
    max_speed_kmph: float = 100.0
    bidirectional: bool = True

    def travel_time_minutes(self, speed_kmph: Optional[float] = None) -> float:
        speed = speed_kmph or self.max_speed_kmph
        speed = max(1e-3, speed)
        return (self.length_km / speed) * 60.0


@dataclass
class RailwayBlock:
    id: str
    section_id: str
    start_km: float
    end_km: float
    headway_minutes: float = 5.0

    def contains(self, position_km: float) -> bool:
        return self.start_km <= position_km <= self.end_km


def build_linear_route(stations: List[Station], default_speed: float = 80.0) -> List[TrackSection]:
    sections: List[TrackSection] = []
    for i in range(len(stations) - 1):
        sections.append(
            TrackSection(
                id=f"S{i+1}",
                from_station=stations[i].code,
                to_station=stations[i + 1].code,
                length_km=50.0,
                max_speed_kmph=default_speed,
            )
        )
    return sections


