"""Evaluation: HR@K and NDCG@K with full-ranking protocol."""

import torch
import numpy as np
from config import cfg


def compute_metrics(
    model,
    edge_index,
    eval_edges,
    train_history,
    num_users,
    num_items,
    k_list=None,
    batch_size=256,
):
    """
    Full-ranking evaluation: score all items per user, mask training items.

    Args:
        model: LightGCN model
        edge_index: undirected training graph edges (for message passing)
        eval_edges: [2, N] edges to evaluate (user->item, unidirectional)
        train_history: dict[user_idx -> set of item_idx]
        num_users: number of users
        num_items: number of items
        k_list: list of K values for HR@K and NDCG@K
        batch_size: number of users per batch
    """
    k_list = k_list or cfg.eval_k
    max_k = max(k_list)
    device = cfg.device
    model.eval()

    # Build ground truth: user -> set of eval item indices
    ground_truth = {}
    src, dst = eval_edges[0].numpy(), eval_edges[1].numpy()
    for u, i in zip(src, dst):
        ground_truth.setdefault(int(u), set()).add(int(i))

    # Only evaluate users that have ground truth
    eval_users = sorted(ground_truth.keys())
    if not eval_users:
        return {f"hr@{k}": 0.0 for k in k_list} | {f"ndcg@{k}": 0.0 for k in k_list}

    # Get all embeddings
    with torch.no_grad():
        emb = model.get_embedding(edge_index.to(device))
        user_emb = emb[:num_users]
        item_emb = emb[num_users:]

    hits = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}

    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start : start + batch_size]
        batch_user_idx = torch.tensor(batch_users, dtype=torch.long, device=device)

        # Score all items: [batch, num_items]
        scores = user_emb[batch_user_idx] @ item_emb.T

        # Mask training items (batch-vectorized)
        row_ids = []
        col_ids = []
        for i, u in enumerate(batch_users):
            history = train_history.get(u, set())
            if history:
                cols = [item - num_users for item in history]
                row_ids.extend([i] * len(cols))
                col_ids.extend(cols)
        if row_ids:
            scores[row_ids, col_ids] = float("-inf")

        # Top-K
        _, topk_indices = scores.topk(max_k, dim=1)
        # Convert back to global item indices
        topk_global = topk_indices + num_users

        for i, u in enumerate(batch_users):
            gt_items = ground_truth[u]
            topk_list = topk_global[i].cpu().tolist()

            for k in k_list:
                topk_k = topk_list[:k]
                # Hit Rate
                hit = 1.0 if any(item in gt_items for item in topk_k) else 0.0
                hits[k].append(hit)

                # NDCG
                dcg = 0.0
                for rank, item in enumerate(topk_k):
                    if item in gt_items:
                        dcg += 1.0 / np.log2(rank + 2)
                ideal_dcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt_items), k)))
                ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
                ndcgs[k].append(ndcg)

    metrics = {}
    for k in k_list:
        metrics[f"hr@{k}"] = np.mean(hits[k])
        metrics[f"ndcg@{k}"] = np.mean(ndcgs[k])
    return metrics
