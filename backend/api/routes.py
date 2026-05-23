"""HTTP routes. Handlers are sync `def` so FastAPI runs them in a threadpool —
this keeps the event loop responsive while PyTorch inference is blocking.
"""

from fastapi import APIRouter, HTTPException, Query

from . import service
from .schemas import (
    Health,
    ItemOut,
    RecommendationOut,
    UserListOut,
    UserOut,
)

router = APIRouter(prefix="/api")


@router.get("/health", response_model=Health)
def health():
    loaded = service.is_loaded()
    nu, ni = service.get_counts() if loaded else (0, 0)
    return Health(status="ok", model_loaded=loaded, num_users=nu, num_items=ni)


@router.get("/users", response_model=UserListOut)
def list_users(
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    q: str | None = Query(None, max_length=128),
):
    total, items = service.list_users(limit=limit, offset=offset, q=q)
    return UserListOut(total=total, items=[UserOut(**i) for i in items])


@router.get("/users/random", response_model=UserOut)
def random_user():
    return UserOut(**service.random_user())


@router.get("/users/{user_idx}", response_model=UserOut)
def get_user(user_idx: int):
    try:
        return UserOut(**service.get_user(user_idx))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/users/{user_idx}/history", response_model=list[ItemOut])
def get_history(user_idx: int, limit: int = Query(50, ge=1, le=500)):
    try:
        return [ItemOut(**i) for i in service.get_history(user_idx, limit)]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/users/{user_idx}/recommendations", response_model=list[RecommendationOut])
def get_recommendations(user_idx: int, k: int = Query(10, ge=1, le=100)):
    try:
        return [RecommendationOut(**i) for i in service.get_recommendations(user_idx, k)]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
