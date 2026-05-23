"""End-to-end pipeline: download -> train. Serving is handled by the FastAPI app."""

import argparse
from config import cfg, logger, seed_everything


def main():
    parser = argparse.ArgumentParser(description="GNN Product Recommender Pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use existing checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--train-only", action="store_true", help="Train only, do not launch anything else")
    parser.add_argument("--legacy-gradio", action="store_true",
                        help="Launch the legacy Gradio demo (deprecated; use the FastAPI server)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    seed_everything(cfg.seed)

    if args.legacy_gradio:
        logger.warning("Launching legacy Gradio demo. Prefer the FastAPI + React frontend.")
        from app_gradio import create_app
        create_app().launch()
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

    logger.info("Done. Start the FastAPI server (separate command) to serve predictions.")


if __name__ == "__main__":
    main()
