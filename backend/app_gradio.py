"""Gradio demo (LEGACY).

Superseded by the FastAPI + React frontend. Kept runnable for ad-hoc inspection
via `python run.py --demo-only`. Do not extend this module — add new endpoints
to the FastAPI service instead.
"""

import torch
import gradio as gr
from config import cfg, logger
from data import preprocess
from model import load_checkpoint


def load_resources():
    """Load model, data, and precompute embeddings."""
    data = preprocess()
    model, ckpt = load_checkpoint()
    model.to(cfg.device)
    model.eval()

    edge_index = data["graph"].edge_index.to(cfg.device)
    with torch.no_grad():
        emb = model.get_embedding(edge_index)
        user_emb = emb[: data["num_users"]].cpu()
        item_emb = emb[data["num_users"] :].cpu()

    return data, user_emb, item_emb


def get_item_title(item_idx, data):
    """Get product title from item index."""
    asin = data["idx_to_item_asin"].get(item_idx, "")
    title = data["asin_to_title"].get(asin, f"Product {asin}")
    return title


def get_user_history_display(user_idx, data):
    """Get formatted purchase history for a user."""
    history = data["train_history"].get(user_idx, set())
    if not history:
        return "No purchase history found."
    items = []
    for item_idx in sorted(history):
        title = get_item_title(item_idx, data)
        items.append(f"- {title}")
    return "\n".join(items[:20])  # Show up to 20 items


def recommend(user_idx, data, user_emb, item_emb):
    """Get top-K recommendations for a user."""
    num_users = data["num_users"]
    scores = user_emb[user_idx] @ item_emb.T
    # Mask training items
    history = data["train_history"].get(user_idx, set())
    for item_idx in history:
        scores[item_idx - num_users] = float("-inf")
    # Top-K
    top_scores, top_indices = scores.topk(cfg.top_k_demo)
    results = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        global_idx = idx + num_users
        title = get_item_title(global_idx, data)
        results.append(f"- **{title}** (score: {score:.3f})")
    return "\n".join(results)


def create_app():
    """Create and return the Gradio app."""
    logger.info("Loading model and data...")
    data, user_emb, item_emb = load_resources()
    num_users = data["num_users"]

    # Sample users that have history for the dropdown
    users_with_history = sorted(
        [u for u in data["train_history"] if len(data["train_history"][u]) >= 3]
    )
    sample_users = users_with_history[: cfg.demo_user_pool_size]

    logger.info(f"Ready! {num_users:,} users, {data['num_items']:,} items")

    def on_recommend(user_choice):
        if user_choice == "Random":
            import random
            user_idx = random.choice(users_with_history)
        else:
            user_idx = int(user_choice)

        original_id = data["idx_to_user_id"].get(user_idx, "unknown")
        history = get_user_history_display(user_idx, data)
        recs = recommend(user_idx, data, user_emb, item_emb)
        info = f"**User Index:** {user_idx} | **Original ID:** {original_id} | **History Size:** {len(data['train_history'].get(user_idx, set()))}"
        return info, history, recs

    user_choices = ["Random"] + [str(u) for u in sample_users]

    with gr.Blocks(title="GNN Product Recommender") as app:
        gr.Markdown("# GNN Product Recommender (Amazon Beauty)")
        gr.Markdown("LightGCN + PyTorch Geometric based recommendation engine")

        with gr.Row():
            user_dropdown = gr.Dropdown(
                choices=user_choices,
                value="Random",
                label="Select User",
                scale=3,
            )
            btn = gr.Button("Recommend", variant="primary", scale=1)

        user_info = gr.Markdown(label="User Info")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Purchase History")
                history_box = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Top-10 Recommendations")
                rec_box = gr.Markdown()

        btn.click(
            fn=on_recommend,
            inputs=[user_dropdown],
            outputs=[user_info, history_box, rec_box],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
