"""Training loop: mini-batch BPR with early stopping."""

import torch
from config import cfg, logger
from data import preprocess
from model import create_model, save_checkpoint
from evaluate import compute_metrics


def train_epoch(model, optimizer, edge_index, train_edges, num_users, num_items, train_history):
    """One epoch of mini-batch BPR training."""
    model.train()
    device = cfg.device
    num_nodes = num_users + num_items

    # Move edge_index to device once
    edge_index = edge_index.to(device)

    # Shuffle training edges
    perm = torch.randperm(train_edges.size(1))
    train_edges_shuffled = train_edges[:, perm]

    total_loss = 0.0
    num_batches = 0

    for start in range(0, train_edges_shuffled.size(1), cfg.batch_size):
        batch = train_edges_shuffled[:, start : start + cfg.batch_size]
        pos_src = batch[0].to(device)
        pos_dst = batch[1].to(device)

        # Negative sampling with false-negative rejection
        neg_dst = torch.randint(num_users, num_nodes, (pos_src.size(0),), device=device)
        src_cpu = pos_src.cpu().tolist()
        neg_cpu = neg_dst.cpu().tolist()
        for i, u in enumerate(src_cpu):
            history = train_history.get(u, set())
            if history and neg_cpu[i] in history:
                for _ in range(10):  # retry up to 10 times
                    candidate = torch.randint(num_users, num_nodes, (1,)).item()
                    if candidate not in history:
                        neg_cpu[i] = candidate
                        break
        neg_dst = torch.tensor(neg_cpu, dtype=torch.long, device=device)

        # Forward: get link predictions
        pos_edge_label_index = torch.stack([pos_src, pos_dst])
        neg_edge_label_index = torch.stack([pos_src, neg_dst])

        pos_rank = model(edge_index, pos_edge_label_index)
        neg_rank = model(edge_index, neg_edge_label_index)

        # BPR loss + L2 regularization (built-in)
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
    # Load data
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

    # Create model or resume from checkpoint
    start_epoch = 1
    best_ndcg = 0.0

    if resume and (cfg.checkpoint_dir / "last_model.pt").exists():
        logger.info("Resuming from last checkpoint...")
        from model import load_checkpoint as load_ckpt
        ckpt_path = cfg.checkpoint_dir / "last_model.pt"
        ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        model = create_model(ckpt["num_nodes"], ckpt["embedding_dim"], ckpt["num_layers"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_ndcg = ckpt.get("best_ndcg", 0.0)
        logger.info(f"Resumed from epoch {ckpt['epoch']}")
    else:
        model = create_model(num_nodes).to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
    )

    epochs = epochs_override or cfg.epochs
    patience_counter = 0
    edge_index = graph.edge_index

    for epoch in range(start_epoch, epochs + 1):
        loss = train_epoch(model, optimizer, edge_index, train_edges, num_users, num_items, train_history)

        # Evaluate periodically
        if epoch % cfg.eval_every == 0 or epoch == start_epoch:
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
                save_checkpoint(model, optimizer, epoch, metrics)
            else:
                patience_counter += cfg.eval_every
                if patience_counter >= cfg.early_stop_patience:
                    logger.info(f"Early stopping at epoch {epoch} (best NDCG@10: {best_ndcg:.4f})")
                    break

            # Save last checkpoint for resume
            save_checkpoint(
                model, optimizer, epoch, metrics,
                path=cfg.checkpoint_dir / "last_model.pt",
            )
        else:
            if epoch <= 10 or epoch % 20 == 0:
                logger.info(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

    # Final test evaluation with best model
    logger.info("--- Test Evaluation ---")
    from model import load_checkpoint as load_ckpt

    best_model, ckpt = load_ckpt()
    best_model.to(cfg.device)
    test_metrics = compute_metrics(
        best_model, edge_index, test_edges, train_history, num_users, num_items
    )
    for k in cfg.eval_k:
        logger.info(f"  HR@{k}: {test_metrics[f'hr@{k}']:.4f}  NDCG@{k}: {test_metrics[f'ndcg@{k}']:.4f}")

    return best_model, test_metrics


if __name__ == "__main__":
    train()
