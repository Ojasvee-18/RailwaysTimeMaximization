from fastapi import APIRouter

router = APIRouter()


@router.get("/ready")
def ready():
    return {"status": "ready"}


@router.get("/live")
def live():
    return {"status": "alive"}


