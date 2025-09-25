from fastapi import APIRouter
from typing import List, Dict
from pydantic import BaseModel
from optimizer.milp_scheduler import MILPScheduler, TrainRequest



class OptimizeRequest(BaseModel):
    headway_minutes: float = 5.0
    trains: List[Dict]


router = APIRouter()


@router.post("/schedule")
def schedule(req: OptimizeRequest) -> Dict[str, List[float]]:
    scheduler = MILPScheduler(headway_minutes=req.headway_minutes)
    requests = [
        TrainRequest(
            train_number=t['train_number'],
            earliest_start_min=float(t.get('earliest_start_min', 0.0)),
            section_times_min=[float(x) for x in t['section_times_min']],
        )
        for t in req.trains
    ]
    return scheduler.schedule(requests)


