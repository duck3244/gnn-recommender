"""Training loop: mini-batch BPR with early stopping."""

import torch
from config import cfg, logger, seed_everything
from data import preprocess
from model import create_model, save_checkpoint, load_checkpoint
from evaluate import compute_metrics


def _build_history_lookup(train_history: dict, num_users: int, num_nodes: int, device):
    """Pack per-user item history as (offsets, items) CSR-style tensors for O(1) GPU rejection.

    Returns:
      offsets: LongTensor[num_users + 1], items: LongTensor[total]
      A user u's items are items[offsets[u]:offsets[u+1]] (global indices, >= num_users).
    """
    offsets = torch.zeros(num_users + 1, dtype=torch.long)
    flat = []
    for u in range(num_users):
        h = train_history.get(u)
        if h:
            flat.extend(h)
        offsets[u + 1] = offsets[u] + (len(h) if h else 0)
    items = torch.tensor(flat, dtype=torch.long) if flat else torch.empty(0, dtype=torch.long)
    return offsets.to(device), items.to(device)


def _sample_negatives(pos_src, num_users, num_nodes, hist_offsets, hist_items, generator):
    """Sample one negative item per row, rejecting items in the user's history.

    Vectorized on-device: each iteration resamples only the rows that are still invalid.
    Falls back after a few rounds with any remaining collisions (rare for sparse graphs).
    """
    device = pos_src.device
    n = pos_src.size(0)
    neg = torch.randint(num_users, num_nodes, (n,), device=device, generator=generator)

    for _ in range(5):
        # Check membership for each row: is neg[i] in items[offsets[u]:offsets[u+1]]?
        starts = hist_offsets[pos_src]
        ends = hist_offsets[pos_src + 1]
        # Build a per-row position in [start, end) where neg appears (or -1).
        # For sparse histories, a small Python-free per-row check: compare
        # against the user's history slice using equality reduction.
        # Use a scatter-style approach: build a per-row boolean by gather + compare.
        bad = torch.zeros(n, dtype=torch.bool, device=device)
        # Max history length determines a padded check; for typical k-core graphs this is small.
        max_len = int((ends - starts).max().item()) if n > 0 else 0
        if max_len == 0:
            return neg
        # arange grid: [n, max_len]
        ar = torch.arange(max_len, device=device).unsqueeze(0).expand(n, max_len)
        idx = starts.unsqueeze(1) + ar  # may go past end; clamp & mask
        valid = ar < (ends - starts).unsqueeze(1)
        idx = idx.clamp(max=hist_items.size(0) - 1)
        gathered = hist_items[idx]  # [n, max_len]
        bad = ((gathered == neg.unsqueeze(1)) & valid).any(dim=1)
        if not bad.any():
            return neg
        resample = torch.randint(num_users, num_nodes, (int(bad.sum().item()),), device=device, generator=generator)
        neg[bad] = resample

    return neg  # accept residual collisions (rare); BPR is robust to a few mislabels


def train_epoch(model, optimizer, edge_index, train_edges, num_users, num_nodes,
                hist_offsets, hist_items, generator):
    """One epoch of mini-batch BPR training."""
    model.train()
    device = cfg.device

    perm = torch.randperm(train_edges.size(1), device=device, generator=generator)
    train_edges_shuffled = train_edges.to(device)[:, perm]

    total_loss = 0.0
    num_batches = 0

    for start in range(0, train_edges_shuffled.size(1), cfg.batch_size):
        batch = train_edges_shuffled[:, start : start + cfg.batch_size]
        pos_src = batch[0]
        pos_dst = batch[1]

        neg_dst = _sample_negatives(
            pos_src, num_users, num_nodes, hist_offsets, hist_items, generator
        )

        # Compute embeddings ONCE per batch (LightGCN.forward would re-run message passing twice).
        emb = model.get_embedding(edge_index)
        src_e = emb[pos_src]
        pos_rank = (src_e * emb[pos_dst]).sum(dim=-1)
        neg_rank = (src_e * emb[neg_dst]).sum(dim=-1)

        involved_nodes = torch.cat([pos_src, pos_dst, neg_dst]).unique()
        loss = model.recommendation_loss(
            pos_rank, neg_rank, node_id=involved_nodes, lambda_reg=cfg.lambda_reg
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train(epochs_override=None, resume=False):
    """Full training pipeline."""
    seed_everything(cfg.seed)

    data = preprocess()
    graph = data["graph"]
    train_edges = data["train_edges"]
    val_edges = data["val_edges"]
    test_edges = data["test_edges"]
    num_users = data["num_users"]
    num_items = data["num_items"]
    train_history = data["train_history"]
    num_nodes = num_users + num_items

    logger.info(f"Device: {cfg.device}")
    logger.info(f"Model: LightGCN (dim={cfg.embedding_dim}, layers={cfg.num_layers})")
    logger.info(f"Nodes: {num_nodes:,}, Train edges: {train_edges.shape[1]:,}")

    start_epoch = 1
    best_ndcg = 0.0
    patience_counter = 0

    # Deterministic generator for shuffling / negative sampling
    generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)

    if resume and (cfg.checkpoint_dir / "last_model.pt").exists():
        logger.info("Resuming from last checkpoint...")
        ckpt_path = cfg.checkpoint_dir / "last_model.pt"
        ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        model = create_model(ckpt["num_nodes"], ckpt["embedding_dim"], ckpt["num_layers"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=cfg.lr_scheduler_factor,
            patience=cfg.lr_scheduler_patience,
        )
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_ndcg = ckpt.get("best_ndcg", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        logger.info(f"Resumed from epoch {ckpt['epoch']} (best NDCG@10: {best_ndcg:.4f})")
    else:
        model = create_model(num_nodes).to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=cfg.lr_scheduler_factor,
            patience=cfg.lr_scheduler_patience,
        )

    epochs = epochs_override or cfg.epochs
    edge_index = graph.edge_index.to(cfg.device)
    hist_offsets, hist_items = _build_history_lookup(train_history, num_users, num_nodes, cfg.device)

    # Early stopping is counted in *evaluation cycles*, not raw epochs.
    eval_patience = max(1, cfg.early_stop_patience // max(1, cfg.eval_every))

    for epoch in range(start_epoch, epochs + 1):
        loss = train_epoch(
            model, optimizer, edge_index, train_edges,
            num_users, num_nodes, hist_offsets, hist_items, generator,
        )

        is_eval_epoch = (epoch % cfg.eval_every == 0) or (epoch == start_epoch) or (epoch == epochs)
        if is_eval_epoch:
            metrics = compute_metrics(
                model, edge_index, val_edges, train_history, num_users, num_items
            )
            ndcg10 = metrics["ndcg@10"]
            hr10 = metrics["hr@10"]
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                f"HR@10: {hr10:.4f} | NDCG@10: {ndcg10:.4f} | LR: {current_lr:.1e}"
            )

            scheduler.step(ndcg10)

            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, epoch, metrics,
                    scheduler=scheduler, best_ndcg=best_ndcg, patience_counter=patience_counter,
                )
            else:
                patience_counter += 1
                if patience_counter >= eval_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"({patience_counter} evals without improvement, best NDCG@10: {best_ndcg:.4f})"
                    )
                    save_checkpoint(
                        model, optimizer, epoch, metrics,
                        path=cfg.checkpoint_dir / "last_model.pt",
                        scheduler=scheduler, best_ndcg=best_ndcg, patience_counter=patience_counter,
                    )
                    break

            save_checkpoint(
                model, optimizer, epoch, metrics,
                path=cfg.checkpoint_dir / "last_model.pt",
                scheduler=scheduler, best_ndcg=best_ndcg, patience_counter=patience_counter,
            )
        elif epoch <= 10 or epoch % 20 == 0:
            logger.info(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

    logger.info("--- Test Evaluation ---")
    best_model, _ = load_checkpoint()
    best_model.to(cfg.device)
    test_metrics = compute_metrics(
        best_model, edge_index, test_edges, train_history, num_users, num_items
    )
    for k in cfg.eval_k:
        logger.info(f"  HR@{k}: {test_metrics[f'hr@{k}']:.4f}  NDCG@{k}: {test_metrics[f'ndcg@{k}']:.4f}")

    return best_model, test_metrics


if __name__ == "__main__":
    train()
