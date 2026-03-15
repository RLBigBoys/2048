from __future__ import annotations

import csv
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rl_proj3.config import Config
from rl_proj3.env import ACTIONS, DOWN, Game2048Env, LEFT, RIGHT, UP
from rl_proj3.features import (
    BlockExtractor,
    BlockExtractor2x3,
    BlockExtractor3x2,
    ColExtractor,
    FeatureExtractor,
    FeatureTuple,
    RowExtractor,
)
from rl_proj3.gui import Pygame2048Viewer


type PolicyArray = np.ndarray
type TaggedFeature = tuple[str, FeatureTuple]
type QTable = dict[TaggedFeature, np.ndarray]
type PolicyTable = dict[TaggedFeature, PolicyArray]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEARNING_CURVE_DIR = PROJECT_ROOT / "learning_curves"
ACTION_LABELS = {
    UP: "Up",
    DOWN: "Down",
    LEFT: "Left",
    RIGHT: "Right",
}


@dataclass(slots=True)
class EpisodeStats:
    """Aggregate metrics for one training episode."""

    reward: float
    steps: int
    score: int
    max_tile: int
    terminated: bool
    truncated: bool


@dataclass(slots=True)
class TrainingSummary:
    """Collection of episode-level statistics."""

    episodes: list[EpisodeStats] = field(default_factory=list)

    def to_learning_curve_rows(self) -> list[dict[str, float | int | bool]]:
        """Serialize per-episode metrics into rows suitable for CSV export."""
        rows: list[dict[str, float | int | bool]] = []
        running_reward = 0.0
        running_score = 0.0

        for episode_index, episode in enumerate(self.episodes, start=1):
            running_reward += episode.reward
            running_score += episode.score
            rows.append(
                {
                    "episode": episode_index,
                    "reward": episode.reward,
                    "score": episode.score,
                    "max_tile": episode.max_tile,
                    "steps": episode.steps,
                    "terminated": episode.terminated,
                    "truncated": episode.truncated,
                    "running_mean_reward": running_reward / episode_index,
                    "running_mean_score": running_score / episode_index,
                }
            )

        return rows

    def save_learning_curve(self, output_path: str | Path) -> Path:
        """Persist per-episode learning metrics to a CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "episode",
            "reward",
            "score",
            "max_tile",
            "steps",
            "terminated",
            "truncated",
            "running_mean_reward",
            "running_mean_score",
        ]

        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.to_learning_curve_rows())

        return output_path

    def save_learning_curve_plot(self, output_path: str | Path) -> Path:
        """Persist learning-curve plots for reward and score as a PNG image."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mpl_config_dir = output_path.parent / ".mplconfig"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = self.to_learning_curve_rows()
        episodes = [int(row["episode"]) for row in rows]
        rewards = [float(row["reward"]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        max_tiles = [float(row["max_tile"]) for row in rows]
        running_rewards = [float(row["running_mean_reward"]) for row in rows]
        running_scores = [float(row["running_mean_score"]) for row in rows]
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        fig.patch.set_facecolor("#f8f6f1")

        panels = (
            (
                axes[0],
                rewards,
                running_rewards,
                "Episode Reward",
                "Reward",
                "#c97b63",
                "#2f5d62",
            ),
            (
                axes[1],
                scores,
                running_scores,
                "Episode Score",
                "Score",
                "#7c9a92",
                "#264653",
            ),
            (
                axes[2],
                max_tiles,
                max_tiles,
                "Max Tile Reached",
                "Max tile",
                "#e9c46a",
                "#bc6c25",
            ),
        )

        for axis, raw_values, smooth_values, title, ylabel, raw_color, mean_color in panels:
            axis.set_facecolor("#fffdf9")
            axis.plot(
                episodes,
                raw_values,
                color=raw_color,
                linewidth=1.2,
                alpha=0.35,
                label="per episode",
            )
            axis.plot(
                episodes,
                smooth_values,
                color=mean_color,
                linewidth=2.2,
                label="running mean" if raw_values is rewards or raw_values is scores else "value",
            )
            axis.set_title(title, fontsize=12)
            axis.set_ylabel(ylabel)
            axis.legend(loc="best")

        axes[2].set_xlabel("Episode")
        fig.suptitle("2048 Training Learning Curves", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output_path, dpi=170)
        plt.close(fig)
        return output_path

    def save_learning_curve_artifacts(self, csv_path: str | Path) -> tuple[Path, Path]:
        """Save both CSV metrics and a PNG plot for the learning curve."""
        csv_path = Path(csv_path)
        plot_path = csv_path.with_suffix(".png")
        return (
            self.save_learning_curve(csv_path),
            self.save_learning_curve_plot(plot_path),
        )

    @property
    def mean_reward(self) -> float:
        """Return the arithmetic mean of episode rewards."""
        if not self.episodes:
            return 0.0
        return float(np.mean([episode.reward for episode in self.episodes]))

    @property
    def mean_score(self) -> float:
        """Return the arithmetic mean of final game scores."""
        if not self.episodes:
            return 0.0
        return float(np.mean([episode.score for episode in self.episodes]))


class LocalMajorityAgent:
    """Tabular local-block agent with configurable global action aggregation."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        policy_tau: float = 0.3,
        seed: int | None = None,
    ) -> None:
        """Initialize Q-tables and the block feature extractor."""
        self.config = config or Config()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_tau = policy_tau
        self.extractors: tuple[tuple[str, FeatureExtractor], ...] = (
            ("block", BlockExtractor(clip_exp=self.config.clip_exp)),
            ("block_2x3", BlockExtractor2x3(clip_exp=self.config.clip_exp)),
            ("block_3x2", BlockExtractor3x2(clip_exp=self.config.clip_exp)),
            ("row", RowExtractor(clip_exp=self.config.clip_exp)),
            ("col", ColExtractor(clip_exp=self.config.clip_exp)),
        )
        self.q_table: QTable = {}
        self.policy_table: PolicyTable = {}
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)

    def extract_features(self, board: np.ndarray) -> list[TaggedFeature]:
        """Encode all configured feature groups for the current board."""
        board_array = board.astype(np.int32, copy=False)
        tagged_features: list[TaggedFeature] = []
        for feature_name, extractor in self.extractors:
            tagged_features.extend(
                (feature_name, feature_tuple)
                for feature_tuple in extractor.extract(board_array)
            )
        return tagged_features

    def _ensure_q_values(self, state: TaggedFeature) -> np.ndarray:
        """Create a zero-initialized Q-row for unseen local states."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS), dtype=np.float64)
        return self.q_table[state]

    def _ensure_policy(self, state: TaggedFeature) -> PolicyArray:
        """Create a uniform policy row for unseen local states."""
        if state not in self.policy_table:
            self.policy_table[state] = np.full(
                len(ACTIONS),
                1.0 / len(ACTIONS),
                dtype=np.float64,
            )
        return self.policy_table[state]

    @staticmethod
    def _masked_argmax(values: np.ndarray, valid_actions: list[int]) -> int:
        """Return the best action among the currently valid global actions."""
        return max(valid_actions, key=lambda action: (float(values[action]), -action))

    @staticmethod
    def _masked_expectation(
        values: np.ndarray,
        policy: PolicyArray,
        valid_actions: list[int],
    ) -> float:
        """Return E_pi[Q(s, a)] over the valid action subset."""
        masked_policy = np.zeros_like(policy)
        masked_policy[valid_actions] = policy[valid_actions]
        total_prob = float(masked_policy.sum())
        if total_prob <= 0.0:
            masked_policy[valid_actions] = 1.0 / len(valid_actions)
        else:
            masked_policy /= total_prob
        return float(np.dot(masked_policy, values))

    def _sample_policy_action(self, state: TaggedFeature, valid_actions: list[int]) -> int:
        """Sample an action from the local policy, restricted to valid moves."""
        policy = self._ensure_policy(state).copy()
        masked_policy = np.zeros_like(policy)
        masked_policy[valid_actions] = policy[valid_actions]
        total_prob = float(masked_policy.sum())
        if total_prob <= 0.0:
            masked_policy[valid_actions] = 1.0 / len(valid_actions)
        else:
            masked_policy /= total_prob
        return int(self.rng.choice(ACTIONS, p=masked_policy))

    def majority_vote_action(
        self,
        board: np.ndarray,
        valid_actions: list[int],
        *,
        use_policy: bool = False,
    ) -> int:
        """Select the global move by aggregating votes from all 2x2 blocks."""
        block_states = self.extract_features(board)
        vote_counts = {action: 0 for action in valid_actions}
        vote_scores = {action: 0.0 for action in valid_actions}

        for state in block_states:
            q_values = self._ensure_q_values(state)
            if use_policy:
                voted_action = self._sample_policy_action(state, valid_actions)
            else:
                voted_action = self._masked_argmax(q_values, valid_actions)

            vote_counts[voted_action] += 1
            vote_scores[voted_action] += float(q_values[voted_action])

        return max(
            valid_actions,
            key=lambda action: (vote_counts[action], vote_scores[action], -action),
        )

    def _block_vote_weight(self, state: TaggedFeature) -> float:
        """Return the importance weight of a local 2x2 block."""
        feature_name, feature_values = state
        max_exp = max(feature_values)
        sum_exp = sum(feature_values)
        feature_bonus = 0.0 if feature_name == "block" else 0.5
        return (
            1.0
            + feature_bonus
            + (self.config.vote_weight_max_tile * float(max_exp))
            + (self.config.vote_weight_sum_tiles * float(sum_exp))
        )

    def weighted_majority_vote_action(
        self,
        board: np.ndarray,
        valid_actions: list[int],
        *,
        use_policy: bool = False,
    ) -> int:
        """Select the global move by weighted voting over all 2x2 blocks."""
        block_states = self.extract_features(board)
        vote_weights = {action: 0.0 for action in valid_actions}
        vote_scores = {action: 0.0 for action in valid_actions}

        for state in block_states:
            q_values = self._ensure_q_values(state)
            if use_policy:
                voted_action = self._sample_policy_action(state, valid_actions)
            else:
                voted_action = self._masked_argmax(q_values, valid_actions)
            block_weight = self._block_vote_weight(state)
            vote_weights[voted_action] += block_weight
            vote_scores[voted_action] += block_weight * float(q_values[voted_action])

        return max(
            valid_actions,
            key=lambda action: (vote_weights[action], vote_scores[action], -action),
        )

    def _aggregate_action(
        self,
        board: np.ndarray,
        valid_actions: list[int],
        *,
        use_policy: bool,
    ) -> int:
        """Choose a global action according to the configured aggregation mode."""
        if use_policy:
            if self.config.action_selection_mode == "majority_vote":
                return self.majority_vote_action(board, valid_actions, use_policy=True)
            return self.weighted_majority_vote_action(board, valid_actions, use_policy=True)
        if self.config.action_selection_mode == "majority_vote":
            return self.majority_vote_action(board, valid_actions, use_policy=False)
        return self.weighted_majority_vote_action(board, valid_actions, use_policy=False)

    def select_action(
        self,
        board: np.ndarray,
        valid_actions: list[int],
        *,
        use_policy: bool = False,
        explore: bool = True,
    ) -> int:
        """Choose the next environment action with epsilon exploration."""
        if not valid_actions:
            raise ValueError("valid_actions must not be empty.")

        if explore and float(self.rng.random()) < self.epsilon:
            return int(self.rng.choice(valid_actions))

        return self._aggregate_action(
            board,
            valid_actions,
            use_policy=use_policy,
        )

    def update_value_iteration(
        self,
        board: np.ndarray,
        action: int,
        reward: float,
        next_board: np.ndarray,
        next_valid_actions: list[int],
        *,
        done: bool,
    ) -> None:
        """Apply a local Q-learning update to all 2x2 blocks."""
        current_states = self.extract_features(board)
        next_states = self.extract_features(next_board)

        for current_state, next_state in zip(current_states, next_states, strict=True):
            current_q = self._ensure_q_values(current_state)
            if done or not next_valid_actions:
                target = reward
            else:
                next_q = self._ensure_q_values(next_state)
                target = reward + self.gamma * max(
                    float(next_q[next_action])
                    for next_action in next_valid_actions
                )
            current_q[action] += self.alpha * (target - float(current_q[action]))

    def policy_evaluation_update(
        self,
        board: np.ndarray,
        action: int,
        reward: float,
        next_board: np.ndarray,
        next_valid_actions: list[int],
        *,
        done: bool,
    ) -> None:
        """Evaluate the current local policy with a TD update on each block."""
        current_states = self.extract_features(board)
        next_states = self.extract_features(next_board)

        for current_state, next_state in zip(current_states, next_states, strict=True):
            current_q = self._ensure_q_values(current_state)
            self._ensure_policy(current_state)

            if done or not next_valid_actions:
                target = reward
            else:
                next_q = self._ensure_q_values(next_state)
                next_policy = self._ensure_policy(next_state)
                target = reward + self.gamma * self._masked_expectation(
                    next_q,
                    next_policy,
                    next_valid_actions,
                )

            current_q[action] += self.alpha * (target - float(current_q[action]))

    def improve_policy(self) -> None:
        """Softly move each local policy toward its greedy Q-policy."""
        for state, q_values in self.q_table.items():
            current_policy = self._ensure_policy(state)
            greedy_action = int(np.argmax(q_values))
            target_policy = np.zeros(len(ACTIONS), dtype=np.float64)
            target_policy[greedy_action] = 1.0
            self.policy_table[state] = (
                (1.0 - self.policy_tau) * current_policy
                + self.policy_tau * target_policy
            )

    def greedy_policy_action(self, board: np.ndarray, valid_actions: list[int]) -> int:
        """Return the deterministic global action from the current local Q-table."""
        return self._aggregate_action(board, valid_actions, use_policy=False)

    def save(self, output_path: str | Path) -> Path:
        """Persist the agent tables to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "policy_tau": self.policy_tau,
            "q_table": {state: values.copy() for state, values in self.q_table.items()},
            "policy_table": {state: values.copy() for state, values in self.policy_table.items()},
        }
        with output_path.open("wb") as file_obj:
            pickle.dump(payload, file_obj)
        return output_path

    @classmethod
    def load(cls, input_path: str | Path) -> LocalMajorityAgent:
        """Restore an agent saved via ``save``."""
        with Path(input_path).open("rb") as file_obj:
            payload = pickle.load(file_obj)

        agent = cls(
            config=payload["config"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            policy_tau=payload["policy_tau"],
            seed=payload["config"].seed,
        )
        agent.q_table = {
            tuple(state): np.array(values, dtype=np.float64)
            for state, values in payload["q_table"].items()
        }
        agent.policy_table = {
            tuple(state): np.array(values, dtype=np.float64)
            for state, values in payload["policy_table"].items()
        }
        return agent


def _run_training_episode(
    env: Game2048Env,
    agent: LocalMajorityAgent,
    *,
    use_policy: bool,
    update_mode: str,
    viewer: Pygame2048Viewer | None = None,
    episode_index: int | None = None,
    total_episodes: int | None = None,
) -> EpisodeStats:
    """Run one full episode and update the local tables online."""
    board, info = env.reset()
    episode_reward = 0.0
    terminated = False
    truncated = False

    if viewer is not None:
        if not viewer.render(
            board,
            info,
            last_reward=0.0,
            terminated=False,
            truncated=False,
            mode_label="Training",
            episode_index=episode_index,
            total_episodes=total_episodes,
            action_label="Reset",
        ):
            return EpisodeStats(
                reward=0.0,
                steps=env.step_count,
                score=env.score,
                max_tile=int(info["max_tile"]),
                terminated=True,
                truncated=False,
            )
        if not viewer.wait(agent.config.visualization_step_delay_ms):
            return EpisodeStats(
                reward=0.0,
                steps=env.step_count,
                score=env.score,
                max_tile=int(info["max_tile"]),
                terminated=True,
                truncated=False,
            )

    while not (terminated or truncated):
        valid_actions = list(info["valid_actions"])
        action = agent.select_action(
            board,
            valid_actions,
            use_policy=use_policy,
            explore=True,
        )
        next_board, reward, terminated, truncated, next_info = env.step(action)

        if update_mode == "value_iteration":
            agent.update_value_iteration(
                board,
                action,
                reward,
                next_board,
                list(next_info["valid_actions"]),
                done=terminated or truncated,
            )
        elif update_mode == "policy_iteration":
            agent.policy_evaluation_update(
                board,
                action,
                reward,
                next_board,
                list(next_info["valid_actions"]),
                done=terminated or truncated,
            )
        else:
            raise ValueError(f"Unknown update_mode: {update_mode}.")

        episode_reward += reward
        board = next_board
        info = next_info

        if viewer is not None:
            if not viewer.render(
                board,
                info,
                last_reward=reward,
                terminated=terminated,
                truncated=truncated,
                mode_label="Training",
                episode_index=episode_index,
                total_episodes=total_episodes,
                action_label=ACTION_LABELS[action],
            ):
                break
            if not viewer.wait(agent.config.visualization_step_delay_ms):
                break

    return EpisodeStats(
        reward=float(episode_reward),
        steps=env.step_count,
        score=env.score,
        max_tile=int(info["max_tile"]),
        terminated=terminated,
        truncated=truncated,
    )


def _set_agent_epsilon_for_progress(
    agent: LocalMajorityAgent,
    *,
    episode_index: int,
    total_episodes: int,
) -> None:
    """Update epsilon according to linear decay configured for training."""
    if not agent.config.use_epsilon_decay:
        agent.epsilon = agent.config.train_epsilon
        return

    if total_episodes <= 1:
        agent.epsilon = agent.config.train_epsilon_end
        return

    progress = (episode_index - 1) / (total_episodes - 1)
    start_epsilon = agent.config.train_epsilon
    end_epsilon = agent.config.train_epsilon_end
    agent.epsilon = start_epsilon + ((end_epsilon - start_epsilon) * progress)


def train_value_iteration(
    num_episodes: int,
    *,
    env: Game2048Env | None = None,
    agent: LocalMajorityAgent | None = None,
    learning_curve_path: str | Path | None = None,
) -> tuple[LocalMajorityAgent, TrainingSummary]:
    """Train with online local Q-learning updates and greedy next-state targets."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    env = env or Game2048Env()
    agent = agent or LocalMajorityAgent(config=env.config)
    summary = TrainingSummary()
    viewer: Pygame2048Viewer | None = None
    if env.config.visualize_training:
        viewer = Pygame2048Viewer(env.config, title="2048 Training")

    for episode_index in range(1, num_episodes + 1):
        _set_agent_epsilon_for_progress(
            agent,
            episode_index=episode_index,
            total_episodes=num_episodes,
        )
        episode_viewer = viewer if (viewer is not None and episode_index % env.config.visualize_training_every_n_episodes == 0) else None
        summary.episodes.append(
            _run_training_episode(
                env,
                agent,
                use_policy=False,
                update_mode="value_iteration",
                viewer=episode_viewer,
                episode_index=episode_index,
                total_episodes=num_episodes,
            )
        )

    curve_path = learning_curve_path or (
        DEFAULT_LEARNING_CURVE_DIR / "value_iteration_learning_curve.csv"
    )
    summary.save_learning_curve_artifacts(curve_path)
    agent.save(env.config.model_path)
    if viewer is not None:
        viewer.close()

    return agent, summary


def train_policy_iteration(
    num_rounds: int,
    episodes_per_round: int,
    *,
    env: Game2048Env | None = None,
    agent: LocalMajorityAgent | None = None,
    learning_curve_path: str | Path | None = None,
) -> tuple[LocalMajorityAgent, TrainingSummary]:
    """Train with alternating policy evaluation and soft greedy improvement."""
    if num_rounds <= 0:
        raise ValueError("num_rounds must be positive.")
    if episodes_per_round <= 0:
        raise ValueError("episodes_per_round must be positive.")

    env = env or Game2048Env()
    agent = agent or LocalMajorityAgent(config=env.config)
    summary = TrainingSummary()
    viewer: Pygame2048Viewer | None = None
    if env.config.visualize_training:
        viewer = Pygame2048Viewer(env.config, title="2048 Training")

    total_episodes = num_rounds * episodes_per_round
    episode_index = 0
    for _ in range(num_rounds):
        for _ in range(episodes_per_round):
            episode_index += 1
            _set_agent_epsilon_for_progress(
                agent,
                episode_index=episode_index,
                total_episodes=total_episodes,
            )
            episode_viewer = viewer if (viewer is not None and episode_index % env.config.visualize_training_every_n_episodes == 0) else None
            summary.episodes.append(
                _run_training_episode(
                    env,
                    agent,
                    use_policy=True,
                    update_mode="policy_iteration",
                    viewer=episode_viewer,
                    episode_index=episode_index,
                    total_episodes=total_episodes,
                )
            )
        agent.improve_policy()

    curve_path = learning_curve_path or (
        DEFAULT_LEARNING_CURVE_DIR / "policy_iteration_learning_curve.csv"
    )
    summary.save_learning_curve_artifacts(curve_path)
    agent.save(env.config.model_path)
    if viewer is not None:
        viewer.close()

    return agent, summary


def train_from_config(
    config: Config | None = None,
) -> tuple[LocalMajorityAgent, TrainingSummary]:
    """Dispatch training according to the method stored in the config."""
    config = config or Config()
    env = Game2048Env(config)
    agent = LocalMajorityAgent(
        config=config,
        alpha=config.train_alpha,
        gamma=config.train_gamma,
        epsilon=config.train_epsilon,
        policy_tau=config.train_policy_tau,
        seed=config.seed,
    )
    learning_curve_dir = config.learning_curve_dir

    if config.training_method == "value_iteration":
        return train_value_iteration(
            config.num_training_episodes,
            env=env,
            agent=agent,
            learning_curve_path=learning_curve_dir / "value_iteration_learning_curve.csv",
        )

    if config.training_method == "policy_iteration":
        return train_policy_iteration(
            config.num_policy_rounds,
            config.episodes_per_policy_round,
            env=env,
            agent=agent,
            learning_curve_path=learning_curve_dir / "policy_iteration_learning_curve.csv",
        )

    raise ValueError(f"Unsupported training_method: {config.training_method}.")
