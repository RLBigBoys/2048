from __future__ import annotations

import os
import sys
import argparse
import logging

ROOT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rl_proj3.config import Config
from rl_proj3.evalu import evaluate_agent, evaluate_saved_agent
from rl_proj3.gui import run_gui
from rl_proj3.train import train_from_config

def _build_arg_parser() -> argparse.ArgumentParser:
    """Create a minimal CLI for selecting the training method."""
    parser = argparse.ArgumentParser(description="Train a local-majority 2048 agent.")
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate", "gui"),
        help="Override Config.run_mode for this run.",
    )
    parser.add_argument(
        "--method",
        choices=("value_iteration", "policy_iteration"),
        help="Override Config.training_method for this run.",
    )
    return parser


def main() -> None:
    """Train the agent configured in Config and print aggregate metrics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _build_arg_parser().parse_args()
    config = Config()

    if args.mode is not None:
        config.run_mode = args.mode
    if args.method is not None:
        config.training_method = args.method
    config.validate()

    if config.run_mode == "gui":
        run_gui(config)
        return

    if config.run_mode == "train":
        logging.info("Starting training: method=%s, episodes=%d", config.training_method, config.num_training_episodes)
        agent, summary = train_from_config(config)
        evaluation = evaluate_agent(
            agent,
            num_episodes=config.num_evaluation_episodes,
            visualize=config.visualize_evaluation,
        )

        print(f"Run mode: {config.run_mode}")
        print(f"Training method: {config.training_method}")
        print(f"Episodes collected: {len(summary.episodes)}")
        print(f"Mean training reward: {summary.mean_reward:.3f}")
        print(f"Mean training score: {summary.mean_score:.3f}")
        print(f"Evaluation mean score: {evaluation.mean_score:.3f}")
        print(f"Evaluation mean max tile: {evaluation.mean_max_tile:.3f}")
        print(
            "Learning curve saved to: "
            f"{config.learning_curve_dir / (config.training_method + '_learning_curve.csv')}"
        )
        print(f"Agent model saved to: {config.model_path}")
        return

    evaluation = evaluate_saved_agent(config)
    print(f"Run mode: {config.run_mode}")
    print(f"Loaded model: {config.model_path}")
    print(f"Evaluation mean score: {evaluation.mean_score:.3f}")
    print(f"Evaluation mean reward: {evaluation.mean_reward:.3f}")
    print(f"Evaluation mean max tile: {evaluation.mean_max_tile:.3f}")
    print(f"Evaluation mean steps: {evaluation.mean_steps:.3f}")


if __name__ == "__main__":
    main()
