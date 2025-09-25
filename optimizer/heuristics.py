"""
Hybrid heuristic scheduler: EDF (Earliest Deadline First) with dynamic
priority weighting and reservation-based conflict resolution.

This replaces the previous FCFS headway heuristic.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class HeuristicRequest:
    train_number: str
    earliest_start_min: float
    section_times_min: List[float]
    # Optional deadlines (start-time targets) per section
    due_times_min: Optional[List[Optional[float]]] = None
    # Static priority (>1.0 gives preference when deadlines tie)
    base_priority: float = 1.0


class HeuristicScheduler:
    def __init__(self, headway_minutes: float = 5.0):
        self.headway = headway_minutes

    def schedule(self, requests: List[HeuristicRequest]) -> Dict[str, List[float]]:
        """
        EDF per-section scheduling with dynamic priority tie-break and
        reservation-based conflict resolution.

        - For each section, build a ready list of trains that need that section
          with their current available time (after finishing previous section).
        - Sort by (deadline, -priority) where deadline = due_time for that
          section if provided, else +inf.
        - Reserve the earliest feasible slot: start at max(train_ready, section_ready)
          and push section_ready forward by processing + headway.
        """
        if not requests:
            return {}

        max_sections = max(len(r.section_times_min) for r in requests)

        # Initialize per-train schedule and availability
        result: Dict[str, List[float]] = {r.train_number: [math.nan] * len(r.section_times_min) for r in requests}
        available_time: Dict[str, float] = {r.train_number: r.earliest_start_min for r in requests}

        # Section reservation state: when the section becomes free next
        section_ready: List[float] = [0.0] * max_sections

        for s_idx in range(max_sections):
            # Build ready list for this section
            ready: List[HeuristicRequest] = [r for r in requests if s_idx < len(r.section_times_min)]

            # Compute effective deadline and priority for sorting
            def sort_key(r: HeuristicRequest):
                due = None
                if r.due_times_min and s_idx < len(r.due_times_min):
                    due = r.due_times_min[s_idx]
                deadline = float(due) if due is not None else float('inf')
                # Earlier deadline first; higher base_priority breaks ties
                return (deadline, -r.base_priority)

            ready.sort(key=sort_key)

            # Reserve slots greedily based on EDF
            for r in ready:
                proc = r.section_times_min[s_idx]
                start = max(available_time[r.train_number], section_ready[s_idx])
                end = start + proc
                # Record reservation
                result[r.train_number][s_idx] = start
                # Update states
                available_time[r.train_number] = end
                section_ready[s_idx] = end + self.headway

        # Replace remaining NaNs with earliest_start (shouldn't happen but safe)
        for r in requests:
            times = result[r.train_number]
            for i, t in enumerate(times):
                if isinstance(t, float) and math.isnan(t):
                    times[i] = available_time[r.train_number]
        return result


