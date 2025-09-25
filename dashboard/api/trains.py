from fastapi import APIRouter, Query
from typing import List, Dict
from datetime import datetime, timedelta
import random

router = APIRouter()


STATIONS = ["NDLS", "BCT", "CSMT", "SDAH", "HWH", "AGC", "CNB", "LKO", "GKP", "PNBE", "BPL", "HBJ", "ET", "JHS", "MAS", "SBC", "HYB", "ADI", "PUNE", "JP"]
TRAIN_NAMES = [
    "Rajdhani Express",
    "Shatabdi Express",
    "Duronto Express",
    "Vande Bharat Express",
    "Tejas Express",
    "Humsafar Express",
    "Garib Rath Express",
    "Jan Shatabdi Express",
    "Sampark Kranti Express",
    "Intercity Express",
    "Punjab Mail",
    "Howrah Mail",
    "Deccan Queen",
    "Konkan Kanya Express",
    "Mandovi Express",
    "Navajivan Express",
    "Tamil Nadu Express",
    "Grand Trunk Express",
    "Kerala Express",
    "Andhra Pradesh Express",
    "Karnataka Express",
    "Azad Hind Express",
    "Gomti Express",
    "Lucknow Mail",
    "Magadh Express",
    "Neelachal Express",
    "Sapt Kranti Express"
]

# Optional: canonical train numbers for popular services (illustrative sample)
# Source numbers are commonly known public train IDs; not exhaustive.
TRAIN_NUMBERS: Dict[str, List[str]] = {
    "Rajdhani Express": ["12951", "12952", "12423", "12424", "22691", "22692"],
    "Shatabdi Express": ["12001", "12002", "12009", "12010", "12029", "12030"],
    "Duronto Express": ["12259", "12260", "12267", "12268"],
    "Vande Bharat Express": ["22435", "22436", "20833", "20834", "20171", "20172"],
    "Tejas Express": ["82901", "82902", "22671", "22672"],
    "Humsafar Express": ["19037", "19038", "22353", "22354"],
    "Garib Rath Express": ["12215", "12216", "12909", "12910"],
    "Jan Shatabdi Express": ["12053", "12054", "12055", "12056"],
    "Sampark Kranti Express": ["12447", "12448", "12649", "12650"],
    "Intercity Express": ["12127", "12128", "12415", "12416"],
    "Punjab Mail": ["12137", "12138"],
    "Howrah Mail": ["12839", "12840"],
    "Deccan Queen": ["12123", "12124"],
    "Konkan Kanya Express": ["10111", "10112"],
    "Mandovi Express": ["10103", "10104"],
    "Navajivan Express": ["12655", "12656"],
    "Tamil Nadu Express": ["12621", "12622"],
    "Grand Trunk Express": ["12615", "12616"],
    "Kerala Express": ["12625", "12626"],
    "Andhra Pradesh Express": ["20805", "20806"],
    "Karnataka Express": ["12627", "12628"],
    "Azad Hind Express": ["12129", "12130"],
    "Gomti Express": ["12419", "12420"],
    "Lucknow Mail": ["12229", "12230"],
    "Magadh Express": ["20801", "20802"],
    "Neelachal Express": ["12875", "12876"],
    "Sapt Kranti Express": ["12557", "12558"]
}


def _build_train(tnum: str) -> Dict:
    tname = random.choice(TRAIN_NAMES)
    route_len = random.randint(3, 6)
    route = random.sample(STATIONS, route_len)
    cur_idx = random.randint(0, route_len - 2)

    # Default stochastic delay; may be overridden by controlled generator
    r = random.random()
    if r < 0.8:
        delay_minutes = random.choice([0, 0, 1, 2, 3])
    elif r < 0.95:
        delay_minutes = random.randint(6, 12)
    else:
        delay_minutes = random.randint(13, 30)

    status = "ON_TIME" if delay_minutes <= 5 else "DELAYED"
    # Prefer canonical number for chosen name when available
    if tname in TRAIN_NUMBERS:
        tnum = random.choice(TRAIN_NUMBERS[tname])

    return {
        "train_number": tnum,
        "train_name": f"{tname}",
        "status": status,
        "delay_minutes": delay_minutes,
        "current_station": route[cur_idx],
        "next_station": route[cur_idx + 1],
        "route": route,
        "last_updated": datetime.utcnow().isoformat()
    }


def _build_train_with_delay(tnum: str, min_delay: int, max_delay: int) -> Dict:
    """Construct a train dict with delay constrained to [min_delay, max_delay]."""
    t = _build_train(tnum)
    # Recompute status based on constrained delay
    d = min_delay if min_delay == max_delay else random.randint(min_delay, max_delay)
    t["delay_minutes"] = max(0, d)
    t["status"] = "ON_TIME" if t["delay_minutes"] <= 5 else "DELAYED"
    return t


@router.get("/live")
def get_live_trains(
    trains: List[str] = Query(default=[]),
    count: int = 50,
    on_time_target: float = None,
) -> List[Dict]:
    """Return live status with on-time share varying between 70% and 98%.

    - If `on_time_target` is provided, it will be clamped to [0.70, 0.98].
    - Otherwise, a random target in [0.70, 0.98] is sampled each call.
    - Single-pass composition (no brute-force retries).
    """
    # Determine target share
    if on_time_target is None:
        target = random.uniform(0.70, 0.98)
    else:
        target = max(0.70, min(on_time_target, 0.98))

    if trains:
        num = len(trains)
        k_on = max(int(num * target + 0.9999), 1)  # ceil
        result: List[Dict] = []
        for i, tn in enumerate(trains):
            if i < k_on:
                result.append(_build_train_with_delay(tn, 0, 3))
            else:
                # Minor/moderate bucket 6-15
                result.append(_build_train_with_delay(tn, 6, 15))
        return result

    base = 12000
    num = max(1, min(count, 200))
    k_on = max(int(num * target + 0.9999), 1)  # ceil
    result: List[Dict] = []
    for i in range(num):
        tnum = str(base + i)
        if i < k_on:
            result.append(_build_train_with_delay(tnum, 0, 3))
        else:
            # mix minor (6-12) and moderate (13-30)
            if (i - k_on) % 5 == 0:
                result.append(_build_train_with_delay(tnum, 13, 30))
            else:
                result.append(_build_train_with_delay(tnum, 6, 12))
    return result


@router.get("/schedule/{train_number}")
def get_schedule(train_number: str) -> Dict:
    """Return a synthetic schedule for the train number."""
    t = _build_train(train_number)
    base_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    stops: List[Dict] = []
    for i, station in enumerate(t["route"]):
        arr = base_time + timedelta(hours=2 * i + random.randint(-1, 1))
        dep = arr + timedelta(minutes=random.randint(2, 8))
        stops.append({
            "station_code": station,
            "station_name": f"Station {station}",
            "arrival_time": arr.isoformat(),
            "departure_time": dep.isoformat(),
            "platform": random.randint(1, 6)
        })
    return {"train_number": train_number, "train_name": t["train_name"], "stops": stops, "status": "OK"}


