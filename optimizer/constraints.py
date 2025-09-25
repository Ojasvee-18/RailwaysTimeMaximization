"""
Safety and signaling constraints helpers.
"""

from dataclasses import dataclass


@dataclass
class SafetyConstraints:
    headway_minutes: float = 5.0
    min_dwell_minutes: float = 1.0


@dataclass
class SignalingRules:
    bidirectional_allowed: bool = True
    max_speed_kmph: float = 120.0


