"""LightGCN model using PyTorch Geometric's built-in implementation."""

import torch
from torch_geometric.nn.models import LightGCN
from config import cfg


def create_model(num_nodes: int, embedding_dim: int = None, num_layers: int = None) -> LightGCN:
    """Create a LightGCN model."""
    return LightGCN(
        num_nodes=num_nodes,
        embedding_dim=embedding_dim or cfg.embedding_dim,
        num_layers=num_layers or cfg.num_layers,
    )


def save_checkpoint(model, optimizer, epoch, metrics, path=None):
    """Save model checkpoint."""
    path = path or cfg.checkpoint_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "num_nodes": model.num_nodes,
            "embedding_dim": model.embedding_dim,
            "num_layers": model.num_layers,
        },
        path,
    )


def load_checkpoint(path=None):
    """Load model from checkpoint."""
    path = path or cfg.checkpoint_dir / "best_model.pt"
    ckpt = torch.load(path, map_location=cfg.device, weights_only=False)
    model = create_model(ckpt["num_nodes"], ckpt["embedding_dim"], ckpt["num_layers"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt
