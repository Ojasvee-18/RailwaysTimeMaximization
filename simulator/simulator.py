"""
Simple discrete-event simulator for trains over linear sections.
"""

from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime

from .blocks import TrackSection
from .train import Train
from .conflict_detector import ConflictDetector


@dataclass
class SimulationResult:
    movements: List[Dict]
    conflicts: List


class RailwaySimulator:
    def __init__(self, sections: List[TrackSection], headway_minutes: float = 5.0):
        self.sections = sections
        self.detector = ConflictDetector(headway_minutes=headway_minutes)

    def simulate(self, trains: List[Train]) -> SimulationResult:
        movements: List[Dict] = []
        for train in trains:
            current_time = train.departure_time
            direction = +1
            for section in self.sections:
                travel_min = section.travel_time_minutes(min(section.max_speed_kmph, train.profile.max_speed_kmph))
                movement = {
                    'train_number': train.number,
                    'section_id': section.id,
                    'enter_min': (current_time - train.departure_time).total_seconds() / 60.0,
                    'exit_min': (current_time - train.departure_time).total_seconds() / 60.0 + travel_min,
                    'direction': direction,
                }
                movements.append(movement)
                current_time = current_time + (current_time - current_time)  # no-op, kept for clarity
                current_time = train.eta_after_distance(section.length_km)
        conflicts = self.detector.detect_headway_conflicts(movements)
        return SimulationResult(movements=movements, conflicts=conflicts)


