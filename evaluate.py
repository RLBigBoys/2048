from __future__ import annotations

import argparse
from pathlib import Path

from rl_proj3.config import Config
from rl_proj3.evalu import evaluate_saved_agent


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create a CLI for evaluating a saved 2048 agent."""
    parser = argparse.ArgumentParser(
        description="Run a saved 2048 agent in evaluation mode.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Override Config.num_evaluation_episodes for this run.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable pygame visualization during evaluation.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Override Config.model_path for this run.",
    )
    return parser


def main() -> None:
    """Load the saved agent and run evaluation episodes."""
    args = _build_arg_parser().parse_args()
    config = Config(run_mode="evaluate")

    if args.episodes is not None:
        config.num_evaluation_episodes = args.episodes
    if args.no_visualize:
        config.visualize_evaluation = False
    if args.model_path is not None:
        config.model_path = args.model_path

    config.validate()

    if not config.model_path.exists():
        raise FileNotFoundError(
            f"Saved agent model not found: {config.model_path}. Run training first.",
        )

    print(f"Evaluating model: {config.model_path}")
    print(f"Episodes: {config.num_evaluation_episodes}")
    print(f"Visualization: {config.visualize_evaluation}")

    evaluation = evaluate_saved_agent(config)

    print(f"Evaluation mean score: {evaluation.mean_score:.3f}")
    print(f"Evaluation mean reward: {evaluation.mean_reward:.3f}")
    print(f"Evaluation mean max tile: {evaluation.mean_max_tile:.3f}")
    print(f"Evaluation mean steps: {evaluation.mean_steps:.3f}")


if __name__ == "__main__":
    main()
