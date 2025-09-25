"""
MILP-based scheduler using pulp (CBC) with priorities, soft lateness, and
per-section headways. Keeps backward compatibility with the simple interface.
"""

from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
import pulp


@dataclass
class TrainRequest:
    train_number: str
    earliest_start_min: float
    section_times_min: List[float]  # processing time per section (minutes)
    # Optional enhanced fields
    release_times_min: Optional[List[float]] = None  # earliest start per section
    due_times_min: Optional[List[Optional[float]]] = None  # desired start per section
    priority_weight: float = 1.0  # higher => more important in objective
    directions: Optional[List[int]] = None  # +1 or -1 per section (optional)


class MILPScheduler:
    def __init__(self, headway_minutes: Union[float, List[float]] = 5.0,
                 lateness_weight: float = 5.0,
                 completion_weight: float = 1.0):
        # Headway can be scalar or per-section list
        self.headway = headway_minutes
        self.lateness_weight = lateness_weight
        self.completion_weight = completion_weight

    def schedule(self, requests: List[TrainRequest]) -> Dict[str, List[float]]:
        # Decision variables: start time of train on each section
        model = pulp.LpProblem("RailwayScheduling", pulp.LpMinimize)

        # Determine max sections across requests
        max_sections = max(len(r.section_times_min) for r in requests)

        # Normalize headway per section
        if isinstance(self.headway, list):
            headway_per_section = self.headway + [self.headway[-1]] * (max_sections - len(self.headway))
        else:
            headway_per_section = [float(self.headway)] * max_sections

        # Variables
        starts: Dict[tuple, pulp.LpVariable] = {}
        lateness: Dict[tuple, pulp.LpVariable] = {}
        for req in requests:
            for s_idx, _ in enumerate(req.section_times_min):
                # Lower bound: either global earliest or per-section release time
                lb = req.earliest_start_min
                if req.release_times_min and s_idx < len(req.release_times_min) and req.release_times_min[s_idx] is not None:
                    lb = max(lb, float(req.release_times_min[s_idx]))
                starts[(req.train_number, s_idx)] = pulp.LpVariable(
                    f"start_{req.train_number}_{s_idx}", lowBound=lb
                )
                # Soft lateness if due_times provided
                due = None
                if req.due_times_min and s_idx < len(req.due_times_min):
                    due = req.due_times_min[s_idx]
                if due is not None:
                    lateness[(req.train_number, s_idx)] = pulp.LpVariable(
                        f"late_{req.train_number}_{s_idx}", lowBound=0.0
                    )

        # Objective: minimize total completion time
        completion_terms = []
        lateness_terms = []
        for req in requests:
            last_idx = len(req.section_times_min) - 1
            comp = starts[(req.train_number, last_idx)] + req.section_times_min[last_idx]
            completion_terms.append(self.completion_weight * req.priority_weight * comp)
            # Sum lateness across sections if defined
            for s_idx in range(len(req.section_times_min)):
                key = (req.train_number, s_idx)
                if key in lateness:
                    lateness_terms.append(self.lateness_weight * req.priority_weight * lateness[key])
        model += pulp.lpSum(completion_terms + lateness_terms)

        # Sequential constraints per train
        for req in requests:
            for s_idx in range(1, len(req.section_times_min)):
                prev = starts[(req.train_number, s_idx - 1)] + req.section_times_min[s_idx - 1]
                model += starts[(req.train_number, s_idx)] >= prev
            # Due time soft constraints
            if req.due_times_min:
                for s_idx in range(len(req.section_times_min)):
                    due = None
                    if s_idx < len(req.due_times_min):
                        due = req.due_times_min[s_idx]
                    if due is not None:
                        # lateness >= start - due
                        model += lateness[(req.train_number, s_idx)] >= starts[(req.train_number, s_idx)] - float(due)
                        # and lateness >= 0 is already enforced

        # Headway constraints same section across trains
        big_m = 1e5
        binaries = {}
        for s_idx in range(max_sections):
            for i in range(len(requests)):
                for j in range(i + 1, len(requests)):
                    a = requests[i]
                    b = requests[j]
                    if s_idx >= len(a.section_times_min) or s_idx >= len(b.section_times_min):
                        continue
                    y_ij = pulp.LpVariable(f"y_{a.train_number}_{b.train_number}_{s_idx}", cat='Binary')
                    binaries[(a.train_number, b.train_number, s_idx)] = y_ij
                    headway = headway_per_section[s_idx]
                    # a before b
                    model += starts[(b.train_number, s_idx)] >= (
                        starts[(a.train_number, s_idx)] + a.section_times_min[s_idx] + headway - big_m * (1 - y_ij)
                    )
                    # b before a
                    model += starts[(a.train_number, s_idx)] >= (
                        starts[(b.train_number, s_idx)] + b.section_times_min[s_idx] + headway - big_m * y_ij
                    )

                    # Optional: direction-based extra separation (if provided)
                    if a.directions and b.directions and s_idx < len(a.directions) and s_idx < len(b.directions):
                        if a.directions[s_idx] != b.directions[s_idx]:
                            # Opposing movement: increase separation by 20%
                            opp_sep = 0.2 * headway
                            model += starts[(b.train_number, s_idx)] >= (
                                starts[(a.train_number, s_idx)] + a.section_times_min[s_idx] + headway + opp_sep - big_m * (1 - y_ij)
                            )
                            model += starts[(a.train_number, s_idx)] >= (
                                starts[(b.train_number, s_idx)] + b.section_times_min[s_idx] + headway + opp_sep - big_m * y_ij
                            )

        # Solve
        model.solve(pulp.PULP_CBC_CMD(msg=False))

        result: Dict[str, List[float]] = {}
        for req in requests:
            times = []
            for s_idx, _ in enumerate(req.section_times_min):
                times.append(pulp.value(starts[(req.train_number, s_idx)]))
            result[req.train_number] = times
        return result


