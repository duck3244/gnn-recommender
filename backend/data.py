"""Data pipeline: download Amazon Beauty, preprocess, build PyG graph."""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from config import cfg, logger


def download_reviews():
    """Download Amazon Beauty reviews from HuggingFace Hub (jsonl)."""
    from huggingface_hub import hf_hub_download
    import json

    logger.info("Downloading Amazon Beauty reviews...")
    local_path = hf_hub_download(
        repo_id=cfg.hf_dataset,
        filename="raw/review_categories/All_Beauty.jsonl",
        repo_type="dataset",
    )
    rows = []
    with open(local_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({
                "user": obj.get("user_id", ""),
                "item": obj.get("parent_asin", obj.get("asin", "")),
                "rating": float(obj.get("rating", 0)),
                "timestamp": int(obj.get("timestamp", 0)),
            })
    df = pd.DataFrame(rows)
    logger.info(f"  Raw reviews: {len(df):,}")
    return df


def download_metadata():
    """Download Amazon Beauty item metadata from HuggingFace Hub (parquet)."""
    from huggingface_hub import hf_hub_download

    logger.info("Downloading Amazon Beauty metadata...")
    local_path = hf_hub_download(
        repo_id=cfg.hf_dataset,
        filename="raw_meta_All_Beauty/full-00000-of-00001.parquet",
        repo_type="dataset",
    )
    meta_df = pd.read_parquet(local_path, columns=["parent_asin", "title"])
    meta_df.columns = ["item", "title"]
    meta_df = meta_df.dropna(subset=["title"])
    meta_df = meta_df.drop_duplicates(subset=["item"])
    asin_to_title = dict(zip(meta_df["item"], meta_df["title"]))
    logger.info(f"  Metadata: {len(asin_to_title):,} items with titles")
    return asin_to_title


def filter_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only positive interactions (rating >= threshold)."""
    df = df[df["rating"] >= cfg.rating_threshold].copy()
    logger.info(f"  After rating >= {cfg.rating_threshold}: {len(df):,} interactions")
    return df


def kcore_filter(df: pd.DataFrame, min_k: int) -> pd.DataFrame:
    """Iterative k-core filtering until convergence."""
    while True:
        user_counts = df["user"].value_counts()
        item_counts = df["item"].value_counts()
        valid_users = user_counts[user_counts >= min_k].index
        valid_items = item_counts[item_counts >= min_k].index
        filtered = df[df["user"].isin(valid_users) & df["item"].isin(valid_items)]
        if len(filtered) == len(df):
            break
        df = filtered.copy()
    n_users = df["user"].nunique()
    n_items = df["item"].nunique()
    logger.info(f"  After {min_k}-core: {n_users:,} users, {n_items:,} items, {len(df):,} interactions")
    return df


def build_mappings(df: pd.DataFrame):
    """Create contiguous ID mappings. Items are offset by num_users."""
    users = sorted(df["user"].unique())
    items = sorted(df["item"].unique())
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: i + len(users) for i, it in enumerate(items)}
    return user_map, item_map, len(users), len(items)


def leave_one_out_split(df: pd.DataFrame):
    """Per-user leave-one-out split: last interaction = test, second-to-last = val."""
    df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)

    # Reverse rank within each user group (1 = last, 2 = second-to-last)
    df["_rank"] = df.groupby("user").cumcount(ascending=False)
    user_counts = df.groupby("user")["user"].transform("count")

    # Users with >= 3 interactions get val/test split
    eligible = user_counts >= 3
    test_mask = eligible & (df["_rank"] == 0)
    val_mask = eligible & (df["_rank"] == 1)
    train_mask = ~test_mask & ~val_mask

    train_df = df.loc[train_mask].drop(columns="_rank").reset_index(drop=True)
    val_df = df.loc[val_mask].drop(columns="_rank").reset_index(drop=True)
    test_df = df.loc[test_mask].drop(columns="_rank").reset_index(drop=True)

    logger.info(f"  Split: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    return train_df, val_df, test_df


def df_to_edge_index(df: pd.DataFrame, user_map: dict, item_map: dict) -> torch.Tensor:
    """Convert DataFrame to edge_index tensor [2, num_edges]."""
    src = torch.tensor([user_map[u] for u in df["user"]], dtype=torch.long)
    dst = torch.tensor([item_map[i] for i in df["item"]], dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def make_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """Add reverse edges to make the graph undirected."""
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def build_user_history(edge_index: torch.Tensor, num_users: int) -> dict:
    """Build a dict mapping user_idx -> set of interacted item_idx (global indices >= num_users).

    Expects a unidirectional user->item edge_index. Item indices must be offset
    by num_users so downstream code can mask via `idx - num_users`.
    """
    history = {}
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    for u, i in zip(src, dst):
        if u < num_users:
            assert i >= num_users, f"history must hold item-space indices, got {i} < {num_users}"
            history.setdefault(int(u), set()).add(int(i))
    return history


def preprocess():
    """Full preprocessing pipeline. Returns processed data dict or loads from cache."""
    cache_path = cfg.processed_dir / "data.pt"
    if cache_path.exists():
        logger.info("Loading cached processed data...")
        return torch.load(cache_path, weights_only=False)

    # Download
    raw_df = download_reviews()
    asin_to_title = download_metadata()

    # Preprocess
    df = filter_positive(raw_df)
    df = kcore_filter(df, cfg.min_interactions)

    # If too few interactions and we asked for stricter filtering, relax to 3-core.
    if len(df) < 10000 and cfg.min_interactions > 3:
        logger.info(f"  Too few interactions at {cfg.min_interactions}-core; relaxing to 3-core...")
        df = filter_positive(raw_df)
        df = kcore_filter(df, 3)

    # Build ID mappings
    user_map, item_map, num_users, num_items = build_mappings(df)
    num_nodes = num_users + num_items
    logger.info(f"  Graph nodes: {num_nodes:,} ({num_users:,} users + {num_items:,} items)")

    # Split
    train_df, val_df, test_df = leave_one_out_split(df)

    # Build edge indices (unidirectional: user -> item)
    train_edges = df_to_edge_index(train_df, user_map, item_map)
    val_edges = df_to_edge_index(val_df, user_map, item_map)
    test_edges = df_to_edge_index(test_df, user_map, item_map)

    # Build undirected graph for message passing (train only)
    train_edge_index = make_undirected(train_edges)

    # User history from training data
    train_history = build_user_history(train_edges, num_users)

    # PyG Data object
    graph = Data(edge_index=train_edge_index, num_nodes=num_nodes)

    # Reverse maps for demo display
    idx_to_item_asin = {v: k for k, v in item_map.items()}
    idx_to_user_id = {v: k for k, v in user_map.items()}

    result = {
        "graph": graph,
        "train_edges": train_edges,
        "val_edges": val_edges,
        "test_edges": test_edges,
        "num_users": num_users,
        "num_items": num_items,
        "train_history": train_history,
        "asin_to_title": asin_to_title,
        "idx_to_item_asin": idx_to_item_asin,
        "idx_to_user_id": idx_to_user_id,
        "user_map": user_map,
        "item_map": item_map,
    }

    # Cache
    torch.save(result, cache_path)
    logger.info(f"  Saved processed data to {cache_path}")
    return result


if __name__ == "__main__":
    data = preprocess()
    logger.info(f"\nSummary:")
    logger.info(f"  Users: {data['num_users']:,}")
    logger.info(f"  Items: {data['num_items']:,}")
    logger.info(f"  Train edges: {data['train_edges'].shape[1]:,}")
    logger.info(f"  Val edges: {data['val_edges'].shape[1]:,}")
    logger.info(f"  Test edges: {data['test_edges'].shape[1]:,}")
    logger.info(f"  Graph edges (undirected): {data['graph'].edge_index.shape[1]:,}")
