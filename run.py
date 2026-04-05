"""End-to-end pipeline: download -> train -> launch demo."""

import argparse
from config import logger


def main():
    parser = argparse.ArgumentParser(description="GNN Product Recommender Pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use existing checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--train-only", action="store_true", help="Train only, don't launch demo")
    parser.add_argument("--demo-only", action="store_true", help="Launch demo only")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    if args.demo_only:
        from app import create_app
        app = create_app()
        app.launch()
        return

    # Step 1: Data
    logger.info("=" * 50)
    logger.info("Step 1: Data Preprocessing")
    logger.info("=" * 50)
    from data import preprocess
    preprocess()

    # Step 2: Train
    if not args.skip_train:
        logger.info("=" * 50)
        logger.info("Step 2: Training")
        logger.info("=" * 50)
        from train import train
        train(epochs_override=args.epochs, resume=args.resume)

    if args.train_only:
        return

    # Step 3: Demo
    logger.info("=" * 50)
    logger.info("Step 3: Launching Gradio Demo")
    logger.info("=" * 50)
    from app import create_app
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
