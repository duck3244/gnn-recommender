"""Pydantic response schemas (snake_case, mirrored 1:1 in the frontend types)."""

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    model_loaded: bool
    num_users: int
    num_items: int


class UserOut(BaseModel):
    user_idx: int
    original_id: str
    history_size: int


class ItemOut(BaseModel):
    item_idx: int
    asin: str
    title: str


class RecommendationOut(BaseModel):
    item_idx: int
    asin: str
    title: str
    score: float


class UserListOut(BaseModel):
    total: int
    items: list[UserOut]


class Error(BaseModel):
    detail: str
    code: str = Field(default="error")
