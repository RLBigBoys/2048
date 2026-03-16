from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_proj3.config import Config
from rl_proj3.env import Game2048Env
from rl_proj3.gui import Pygame2048Viewer
from rl_proj3.train import LocalMajorityAgent


@dataclass(slots=True)
class EvaluationStats:
    """Aggregate metrics collected over evaluation episodes."""

    mean_score: float
    mean_reward: float
    mean_max_tile: float
    mean_steps: float


def evaluate_agent(
    agent: LocalMajorityAgent,
    num_episodes: int,
    *,
    env: Game2048Env | None = None,
    visualize: bool = False,
) -> EvaluationStats:
    """Run greedy evaluation episodes without exploration or learning."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    env = env or Game2048Env(agent.config)
    viewer: Pygame2048Viewer | None = None
    if visualize:
        viewer = Pygame2048Viewer(agent.config, title="2048 Agent Evaluation")
    scores: list[int] = []
    rewards: list[float] = []
    max_tiles: list[int] = []
    steps: list[int] = []

    for episode_index in range(1, num_episodes + 1):
        board, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        if viewer is not None:
            if not viewer.render(
                board,
                info,
                last_reward=0.0,
                terminated=False,
                truncated=False,
                mode_label="Evaluation",
                episode_index=episode_index,
                total_episodes=num_episodes,
                action_label="Reset",
            ):
                break
            if not viewer.wait(agent.config.visualization_step_delay_ms):
                break

        while not (terminated or truncated):
            action = agent.greedy_policy_action(board, list(info["valid_actions"]))
            board, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if viewer is not None:
                if not viewer.render(
                    board,
                    info,
                    last_reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    mode_label="Evaluation",
                    episode_index=episode_index,
                    total_episodes=num_episodes,
                    action_label={0: "Up", 1: "Down", 2: "Left", 3: "Right"}[action],
                ):
                    terminated = True
                    truncated = True
                    break
                if not viewer.wait(agent.config.visualization_step_delay_ms):
                    terminated = True
                    truncated = True
                    break

        scores.append(env.score)
        rewards.append(total_reward)
        max_tiles.append(int(info["max_tile"]))
        steps.append(env.step_count)

    if viewer is not None:
        viewer.close()

    if not scores:
        return EvaluationStats(
            mean_score=0.0,
            mean_reward=0.0,
            mean_max_tile=0.0,
            mean_steps=0.0,
        )

    return EvaluationStats(
        mean_score=float(np.mean(scores)),
        mean_reward=float(np.mean(rewards)),
        mean_max_tile=float(np.mean(max_tiles)),
        mean_steps=float(np.mean(steps)),
    )


def evaluate_saved_agent(config: Config | None = None) -> EvaluationStats:
    """Load the persisted agent and run evaluation according to the config."""
    config = config or Config()
    agent = LocalMajorityAgent.load(config.model_path)
    env = Game2048Env(config)
    return evaluate_agent(
        agent,
        config.num_evaluation_episodes,
        env=env,
        visualize=config.visualize_evaluation,
    )
