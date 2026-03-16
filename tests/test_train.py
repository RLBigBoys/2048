from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_proj3.config import Config
from rl_proj3.env import Game2048Env
from rl_proj3.env import LEFT, RIGHT
from rl_proj3.evalu import evaluate_saved_agent
from rl_proj3.train import LocalMajorityAgent, train_from_config, train_value_iteration


def _make_unique_board() -> np.ndarray:
    return np.array(
        [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [8192, 16384, 32768, 65536],
        ],
        dtype=np.int32,
    )


def test_majority_vote_action_uses_block_majority() -> None:
    agent = LocalMajorityAgent(
        config=Config(clip_exp=16),
        epsilon=0.0,
    )
    board = _make_unique_board()
    states = agent.extract_features(board)

    for state in states:
        feature_name, _ = state
        q_values = agent._ensure_q_values(state)
        if feature_name in {"block", "block_2x3"}:
            q_values[LEFT] = 3.0
            q_values[RIGHT] = 1.0
        else:
            q_values[LEFT] = 1.0
            q_values[RIGHT] = 3.0

    action = agent.majority_vote_action(board, [LEFT, RIGHT])

    assert action == LEFT


def test_weighted_majority_vote_prefers_more_important_blocks() -> None:
    agent = LocalMajorityAgent(
        config=Config(
            clip_exp=16,
            action_selection_mode="weighted_majority_vote",
            vote_weight_max_tile=2.0,
            vote_weight_sum_tiles=0.2,
        ),
        epsilon=0.0,
    )
    board = _make_unique_board()
    states = agent.extract_features(board)

    for state in states:
        feature_name, feature_values = state
        q_values = agent._ensure_q_values(state)
        if max(feature_values) >= 15 or feature_name != "block":
            q_values[LEFT] = 1.0
            q_values[RIGHT] = 4.0
        else:
            q_values[LEFT] = 3.0
            q_values[RIGHT] = 1.0

    action = agent.weighted_majority_vote_action(board, [LEFT, RIGHT])

    assert action == RIGHT


def test_greedy_policy_action_uses_configured_weighted_vote_mode() -> None:
    agent = LocalMajorityAgent(
        config=Config(
            clip_exp=16,
            action_selection_mode="weighted_majority_vote",
            vote_weight_max_tile=2.0,
            vote_weight_sum_tiles=0.2,
        ),
        epsilon=0.0,
    )
    board = _make_unique_board()
    states = agent.extract_features(board)

    for state in states:
        feature_name, feature_values = state
        q_values = agent._ensure_q_values(state)
        if max(feature_values) >= 15 or feature_name != "block":
            q_values[LEFT] = 1.0
            q_values[RIGHT] = 4.0
        else:
            q_values[LEFT] = 3.0
            q_values[RIGHT] = 1.0

    action = agent.greedy_policy_action(board, [LEFT, RIGHT])

    assert action == RIGHT


def test_update_value_iteration_updates_every_local_block() -> None:
    agent = LocalMajorityAgent(
        config=Config(clip_exp=16),
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
    )
    board = _make_unique_board()
    next_board = np.rot90(board)
    states = agent.extract_features(board)

    agent.update_value_iteration(
        board,
        LEFT,
        reward=2.0,
        next_board=next_board,
        next_valid_actions=[LEFT, RIGHT],
        done=True,
    )

    for state in states:
        assert agent.q_table[state][LEFT] == pytest.approx(1.0)


def test_policy_evaluation_update_uses_expected_next_value() -> None:
    agent = LocalMajorityAgent(
        config=Config(clip_exp=16),
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
    )
    board = _make_unique_board()
    next_board = np.array(
        [
            [0, 2, 0, 4],
            [8, 0, 16, 0],
            [0, 32, 0, 64],
            [128, 0, 256, 0],
        ],
        dtype=np.int32,
    )
    next_states = agent.extract_features(next_board)

    for state in next_states:
        q_values = agent._ensure_q_values(state)
        q_values[LEFT] = 2.0
        q_values[RIGHT] = 4.0
        policy = agent._ensure_policy(state)
        policy[:] = 0.0
        policy[LEFT] = 0.25
        policy[RIGHT] = 0.75

    expected_bootstrap = 0.25 * 2.0 + 0.75 * 4.0
    expected_target = 1.0 + (0.9 * expected_bootstrap)
    expected_update = 0.5 * expected_target

    agent.policy_evaluation_update(
        board,
        LEFT,
        reward=1.0,
        next_board=next_board,
        next_valid_actions=[LEFT, RIGHT],
        done=False,
    )

    for state in agent.extract_features(board):
        assert agent.q_table[state][LEFT] == pytest.approx(expected_update)


def test_agent_extracts_blocks_rows_and_columns() -> None:
    agent = LocalMajorityAgent(config=Config(clip_exp=16))
    board = _make_unique_board()

    states = agent.extract_features(board)
    feature_names = [feature_name for feature_name, _ in states]

    assert len(states) == 29
    assert feature_names.count("block") == 9
    assert feature_names.count("block_2x3") == 6
    assert feature_names.count("block_3x2") == 6
    assert feature_names.count("row") == 4
    assert feature_names.count("col") == 4


def test_train_value_iteration_saves_learning_curve_csv(tmp_path: Path) -> None:
    env = Game2048Env(Config(max_steps_per_episode=4, spawn_prob_2=1.0, spawn_prob_4=0.0))
    agent = LocalMajorityAgent(
        config=env.config,
        epsilon=0.0,
        seed=0,
    )
    output_path = tmp_path / "curves" / "value_iteration_learning_curve.csv"

    _, summary = train_value_iteration(
        2,
        env=env,
        agent=agent,
        learning_curve_path=output_path,
    )

    assert output_path.exists()
    assert output_path.with_suffix(".png").exists()

    with output_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 2
    assert rows[0]["episode"] == "1"
    assert rows[-1]["episode"] == "2"
    assert float(rows[-1]["running_mean_reward"]) == pytest.approx(summary.mean_reward)


def test_train_from_config_uses_selected_method(tmp_path: Path) -> None:
    config = Config(
        training_method="policy_iteration",
        num_policy_rounds=1,
        episodes_per_policy_round=2,
        max_steps_per_episode=4,
        spawn_prob_2=1.0,
        spawn_prob_4=0.0,
        model_path=tmp_path / "agent.pkl",
        learning_curve_dir=tmp_path,
    )

    _, summary = train_from_config(config)

    assert len(summary.episodes) == 2
    assert (tmp_path / "policy_iteration_learning_curve.csv").exists()
    assert (tmp_path / "agent.pkl").exists()


def test_saved_agent_can_be_loaded_for_evaluation(tmp_path: Path) -> None:
    config = Config(
        training_method="value_iteration",
        num_training_episodes=2,
        num_evaluation_episodes=1,
        max_steps_per_episode=4,
        spawn_prob_2=1.0,
        spawn_prob_4=0.0,
        visualize_evaluation=False,
        model_path=tmp_path / "agent.pkl",
        learning_curve_dir=tmp_path,
    )

    train_from_config(config)
    evaluation = evaluate_saved_agent(config)

    assert evaluation.mean_steps >= 0.0


def test_train_from_config_applies_epsilon_decay(tmp_path: Path) -> None:
    config = Config(
        training_method="value_iteration",
        num_training_episodes=4,
        max_steps_per_episode=3,
        spawn_prob_2=1.0,
        spawn_prob_4=0.0,
        train_epsilon=0.2,
        train_epsilon_end=0.05,
        use_epsilon_decay=True,
        visualize_evaluation=False,
        model_path=tmp_path / "agent.pkl",
        learning_curve_dir=tmp_path,
    )

    agent, _ = train_from_config(config)

    assert agent.epsilon == pytest.approx(config.train_epsilon_end)


def test_train_from_config_saves_checkpoints(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    config = Config(
        training_method="value_iteration",
        num_training_episodes=3,
        max_steps_per_episode=3,
        spawn_prob_2=1.0,
        spawn_prob_4=0.0,
        visualize_evaluation=False,
        model_path=tmp_path / "agent.pkl",
        learning_curve_dir=tmp_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_n_episodes=1,
        save_checkpoints=True,
    )

    train_from_config(config)

    checkpoint_files = sorted(checkpoint_dir.glob("agent_episode_*.pkl"))
    assert len(checkpoint_files) == 3
    assert checkpoint_files[0].name == "agent_episode_000001.pkl"
    assert checkpoint_files[-1].name == "agent_episode_000003.pkl"


def test_train_from_config_saves_best_checkpoint(tmp_path: Path) -> None:
    best_model_path = tmp_path / "best_agent.pkl"
    config = Config(
        training_method="value_iteration",
        num_training_episodes=3,
        max_steps_per_episode=3,
        spawn_prob_2=1.0,
        spawn_prob_4=0.0,
        visualize_evaluation=False,
        model_path=tmp_path / "agent.pkl",
        learning_curve_dir=tmp_path,
        save_checkpoints=False,
        save_best_checkpoint=True,
        best_checkpoint_metric="max_tile",
        best_model_path=best_model_path,
    )

    train_from_config(config)

    assert best_model_path.exists()


def test_train_from_config_can_resume_from_saved_model(tmp_path: Path) -> None:
    base_model_path = tmp_path / "agent.pkl"
    initial_config = Config(
        training_method="value_iteration",
        num_training_episodes=2,
        max_steps_per_episode=3,
        spawn_prob_2=1.0,
        spawn_prob_4=0.0,
        visualize_evaluation=False,
        model_path=base_model_path,
        learning_curve_dir=tmp_path,
        save_checkpoints=False,
        save_best_checkpoint=False,
    )

    train_from_config(initial_config)
    initial_agent = LocalMajorityAgent.load(base_model_path)
    initial_q_table_size = len(initial_agent.q_table)

    resumed_config = Config(
        training_method="value_iteration",
        num_training_episodes=1,
        max_steps_per_episode=3,
        spawn_prob_2=1.0,
        spawn_prob_4=0.0,
        visualize_evaluation=False,
        model_path=base_model_path,
        learning_curve_dir=tmp_path,
        save_checkpoints=False,
        save_best_checkpoint=False,
        resume_training=True,
        resume_model_path=base_model_path,
    )

    resumed_agent, summary = train_from_config(resumed_config)

    assert len(summary.episodes) == 1
    assert len(resumed_agent.q_table) >= initial_q_table_size
