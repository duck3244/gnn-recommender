import os
import logging
import torch
import numpy as np
import random
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

@dataclass
class Config:
    # Dataset
    hf_dataset: str = "McAuley-Lab/Amazon-Reviews-2023"
    min_interactions: int = 3
    rating_threshold: float = 3.0

    # Model
    embedding_dim: int = 64
    num_layers: int = 3

    # Training
    lr: float = 1e-3
    lambda_reg: float = 1e-4
    batch_size: int = 4096
    epochs: int = 200
    early_stop_patience: int = 20
    eval_every: int = 5
    seed: int = 42
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5

    # Evaluation
    eval_k: list = field(default_factory=lambda: [10, 20])

    # Demo
    top_k_demo: int = 10

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    checkpoint_dir: Path = PROJECT_ROOT / "data" / "checkpoints"

    # Device
    device: torch.device = field(default=None)

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for d in [self.raw_dir, self.processed_dir, self.checkpoint_dir]:
            d.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(level=logging.INFO):
    """Configure logging for the project."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    return logging.getLogger("gnn-recommender")


cfg = Config()
logger = setup_logging()
seed_everything(cfg.seed)
