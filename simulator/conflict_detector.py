"""
Detect headway and opposing-movement conflicts.
"""

from typing import List, Dict, Tuple


class ConflictDetector:
    def __init__(self, headway_minutes: float = 5.0):
        self.headway_minutes = headway_minutes

    def detect_headway_conflicts(self, movements: List[Dict]) -> List[Tuple[Dict, Dict]]:
        # movements: [{section_id, enter_min, exit_min, direction, train_number}]
        conflicts = []
        by_section: Dict[str, List[Dict]] = {}
        for m in movements:
            by_section.setdefault(m['section_id'], []).append(m)

        for section_id, items in by_section.items():
            items.sort(key=lambda x: x['enter_min'])
            for i in range(1, len(items)):
                prev = items[i - 1]
                curr = items[i]
                # Same direction headway
                if prev['direction'] == curr['direction']:
                    if curr['enter_min'] - prev['enter_min'] < self.headway_minutes:
                        conflicts.append((prev, curr))
                # Opposite direction overlap
                else:
                    if not (curr['enter_min'] >= prev['exit_min'] or prev['enter_min'] >= curr['exit_min']):
                        conflicts.append((prev, curr))
        return conflicts


