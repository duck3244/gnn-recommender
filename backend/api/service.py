"""Inference service: load model + data once, expose recommendation helpers.

Single-user MVP: all state is module-level and immutable after `load()`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch

from config import cfg, logger
from data import preprocess
from model import load_checkpoint


@dataclass
class _State:
    data: dict
    user_emb: torch.Tensor  # [num_users, D] on CPU
    item_emb: torch.Tensor  # [num_items, D] on CPU
    num_users: int
    num_items: int


_state: Optional[_State] = None


def load() -> None:
    """Load preprocessed data + best checkpoint, precompute embeddings on CPU."""
    global _state
    logger.info("Loading data + model for serving...")

    data = preprocess()
    model, ckpt = load_checkpoint()

    # Sanity: checkpoint and current data must agree on graph size.
    expected_nodes = data["num_users"] + data["num_items"]
    if ckpt["num_nodes"] != expected_nodes:
        raise RuntimeError(
            f"Checkpoint num_nodes ({ckpt['num_nodes']}) != data num_nodes "
            f"({expected_nodes}). Re-train or restore the matching dataset."
        )

    # Serving runs on CPU for predictability (single user, fast enough).
    model.to("cpu")
    model.eval()

    edge_index = data["graph"].edge_index  # already CPU
    with torch.no_grad():
        emb = model.get_embedding(edge_index)
    num_users = data["num_users"]

    _state = _State(
        data=data,
        user_emb=emb[:num_users].contiguous(),
        item_emb=emb[num_users:].contiguous(),
        num_users=num_users,
        num_items=data["num_items"],
    )
    logger.info(
        f"Service ready: {_state.num_users:,} users, {_state.num_items:,} items"
    )


def is_loaded() -> bool:
    return _state is not None


def _require() -> _State:
    if _state is None:
        raise RuntimeError("Service not loaded. Call service.load() first.")
    return _state


def get_counts() -> tuple[int, int]:
    s = _require()
    return s.num_users, s.num_items


def _user_summary(user_idx: int) -> dict:
    s = _require()
    if not (0 <= user_idx < s.num_users):
        raise KeyError(f"user_idx out of range: {user_idx}")
    return {
        "user_idx": user_idx,
        "original_id": str(s.data["idx_to_user_id"].get(user_idx, "")),
        "history_size": len(s.data["train_history"].get(user_idx, set())),
    }


def list_users(limit: int, offset: int, q: str | None = None) -> tuple[int, list[dict]]:
    s = _require()
    # Only users that have at least one interaction in train (display candidates).
    pool = [u for u in range(s.num_users) if s.data["train_history"].get(u)]
    if q:
        ql = q.lower()
        idx_to_id = s.data["idx_to_user_id"]
        pool = [u for u in pool if ql in str(idx_to_id.get(u, "")).lower()]
    total = len(pool)
    sliced = pool[offset : offset + limit]
    return total, [_user_summary(u) for u in sliced]


def get_user(user_idx: int) -> dict:
    return _user_summary(user_idx)


def random_user() -> dict:
    s = _require()
    pool = [u for u in range(s.num_users) if s.data["train_history"].get(u)]
    if not pool:
        raise RuntimeError("No users with history available")
    return _user_summary(random.choice(pool))


def _item_payload(global_item_idx: int) -> dict:
    s = _require()
    asin = str(s.data["idx_to_item_asin"].get(global_item_idx, ""))
    title = str(s.data["asin_to_title"].get(asin, f"Product {asin}"))
    return {"item_idx": global_item_idx, "asin": asin, "title": title}


def get_history(user_idx: int, limit: int) -> list[dict]:
    s = _require()
    if not (0 <= user_idx < s.num_users):
        raise KeyError(f"user_idx out of range: {user_idx}")
    history = sorted(s.data["train_history"].get(user_idx, set()))
    return [_item_payload(i) for i in history[:limit]]


def get_recommendations(user_idx: int, k: int) -> list[dict]:
    s = _require()
    if not (0 <= user_idx < s.num_users):
        raise KeyError(f"user_idx out of range: {user_idx}")
    if k <= 0:
        return []

    scores = s.user_emb[user_idx] @ s.item_emb.T  # [num_items]
    # Mask training items
    history = s.data["train_history"].get(user_idx, set())
    if history:
        # history holds global indices >= num_users
        mask_cols = torch.tensor(
            [i - s.num_users for i in history], dtype=torch.long
        )
        scores = scores.clone()
        scores[mask_cols] = float("-inf")

    k = min(k, s.num_items)
    top_scores, top_idx_local = scores.topk(k)
    out = []
    for sc, local in zip(top_scores.tolist(), top_idx_local.tolist()):
        payload = _item_payload(local + s.num_users)
        payload["score"] = float(sc)
        out.append(payload)
    return out
