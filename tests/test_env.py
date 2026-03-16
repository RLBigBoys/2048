from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_proj3.config import Config
from rl_proj3.env import DOWN, LEFT, RIGHT, UP, Game2048Env


@pytest.mark.parametrize(
    ("line_values", "expected_line", "expected_gain"),
    [
        ([2, 2, 2, 0], [4, 2, 0, 0], 4),
        ([4, 0, 2, 2], [4, 4, 0, 0], 4),
        ([2, 2, 4, 4], [4, 8, 0, 0], 12),
    ],
)
def test_compress_and_merge_line(
    line_values: list[int],
    expected_line: list[int],
    expected_gain: int,
) -> None:
    env = Game2048Env(Config())
    merged_line, merge_gain = env._compress_and_merge_line(
        np.array(line_values, dtype=np.int32),
    )

    assert merged_line.tolist() == expected_line
    assert merge_gain == expected_gain


def test_invalid_move_keeps_board_unchanged_and_does_not_spawn() -> None:
    env = Game2048Env(Config(spawn_prob_2=1.0, spawn_prob_4=0.0))
    env.reset(seed=7)
    env.board[:, :] = np.array(
        [
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    board_before = env.board.copy()
    non_zero_before = int(np.count_nonzero(board_before))

    next_board, reward, terminated, truncated, info = env.step(LEFT)

    assert np.array_equal(next_board, board_before)
    assert int(np.count_nonzero(next_board)) == non_zero_before
    assert reward == env.config.invalid_move_penalty
    assert terminated is False
    assert truncated is False
    assert info["changed"] is False
    assert info["merge_gain"] == 0
    assert info["delta_empty"] == 0
    assert info["delta_max_exp"] == 0


def test_terminal_board_is_detected_when_no_actions_remain() -> None:
    env = Game2048Env(Config())
    env.reset(seed=11)
    env.board[:, :] = np.array(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ],
        dtype=np.int32,
    )

    assert env.get_valid_actions() == []

    next_board, reward, terminated, truncated, info = env.step(UP)

    assert np.array_equal(next_board, env.board)
    assert reward == env.config.invalid_move_penalty
    assert terminated is True
    assert truncated is False
    assert info["valid_actions"] == []
    assert info["action_mask"].tolist() == [0, 0, 0, 0]


def test_valid_actions_match_expected_for_simple_board() -> None:
    env = Game2048Env(Config())
    env.reset(seed=5)
    env.board[:, :] = np.array(
        [
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    valid_actions = env.get_valid_actions()

    assert valid_actions == [DOWN, LEFT, RIGHT]


def test_reward_uses_pre_spawn_board_deltas() -> None:
    env = Game2048Env(
        Config(
            spawn_prob_2=1.0,
            spawn_prob_4=0.0,
            reward_max_tile_in_corner_bonus=0.0,
            reward_max_tile_out_of_corner_penalty=0.0,
        )
    )
    env.reset(seed=13)
    env.board[:, :] = np.array(
        [
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    next_board, reward, terminated, truncated, info = env.step(LEFT)

    assert np.count_nonzero(next_board) == 2
    assert reward == pytest.approx(0.0)
    assert terminated is False
    assert truncated is False
    assert info["changed"] is True
    assert info["merge_gain"] == 0
    assert info["delta_empty"] == 0
    assert info["delta_max_exp"] == 0


def test_reaching_target_tile_adds_bonus_and_terminates() -> None:
    env = Game2048Env(
        Config(
            target_tile=2048,
            spawn_prob_2=1.0,
            spawn_prob_4=0.0,
            terminate_on_target_tile=True,
            reward_target_tile_bonus=500.0,
            reward_max_tile_in_corner_bonus=0.0,
            reward_max_tile_out_of_corner_penalty=0.0,
        )
    )
    env.reset(seed=17)
    env.board[:, :] = np.array(
        [
            [1024, 1024, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    next_board, reward, terminated, truncated, info = env.step(LEFT)

    assert terminated is True
    assert truncated is False
    assert info["reached_target_tile"] is True
    assert int(np.max(next_board)) >= 2048
    expected_reward = (
        (2048 * env.config.reward_score_scale)
        + (1 * env.config.reward_empty_bonus)
        + (1 * env.config.reward_max_tile_bonus)
        + env.config.reward_target_tile_bonus
    )
    assert reward == pytest.approx(expected_reward)


def test_reaching_target_tile_can_continue_when_configured() -> None:
    env = Game2048Env(
        Config(
            target_tile=2048,
            spawn_prob_2=1.0,
            spawn_prob_4=0.0,
            terminate_on_target_tile=False,
            reward_target_tile_bonus=200.0,
        )
    )
    env.reset(seed=19)
    env.board[:, :] = np.array(
        [
            [1024, 1024, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    _, reward, terminated, truncated, info = env.step(LEFT)

    assert terminated is False
    assert truncated is False
    assert info["reached_target_tile"] is True
    assert reward >= env.config.reward_target_tile_bonus


def test_stagnation_penalty_applies_after_configured_threshold() -> None:
    env = Game2048Env(
        Config(
            spawn_prob_2=1.0,
            spawn_prob_4=0.0,
            reward_stagnation_penalty=0.25,
            stagnation_penalty_after_steps=3,
        )
    )
    env.reset(seed=23)
    env.board[:, :] = np.array(
        [
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    _, reward_1, _, _, info_1 = env.step(LEFT)
    _, reward_2, _, _, info_2 = env.step(LEFT)
    _, reward_3, _, _, info_3 = env.step(LEFT)

    assert reward_1 == env.config.invalid_move_penalty
    assert reward_2 == env.config.invalid_move_penalty
    assert reward_3 == pytest.approx(
        env.config.invalid_move_penalty - env.config.reward_stagnation_penalty,
    )
    assert info_1["stagnation_penalty_applied"] is False
    assert info_2["stagnation_penalty_applied"] is False
    assert info_3["stagnation_penalty_applied"] is True
    assert info_3["stagnation_steps"] == 3


def test_stagnation_counter_resets_after_max_tile_progress() -> None:
    env = Game2048Env(
        Config(
            spawn_prob_2=1.0,
            spawn_prob_4=0.0,
            reward_stagnation_penalty=0.25,
            stagnation_penalty_after_steps=2,
        )
    )
    env.reset(seed=29)
    env.board[:, :] = np.array(
        [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    _, _, _, _, info_before = env.step(LEFT)
    env.board[:, :] = np.array(
        [
            [4, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    _, _, _, _, info_after = env.step(LEFT)

    assert info_before["delta_max_exp"] == 1
    assert info_before["stagnation_steps"] == 0
    assert info_after["stagnation_steps"] == 1


def test_corner_bonus_applies_when_max_tile_is_in_target_corner() -> None:
    env = Game2048Env(
        Config(
            target_corner="top_left",
            reward_max_tile_in_corner_bonus=0.5,
            reward_max_tile_out_of_corner_penalty=0.1,
            spawn_prob_2=1.0,
            spawn_prob_4=0.0,
        )
    )
    env.reset(seed=31)
    env.board[:, :] = np.array(
        [
            [4, 0, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    _, reward, _, _, info = env.step(RIGHT)

    assert info["max_tile_in_target_corner"] is False
    assert reward <= 0.0


def test_corner_reward_detects_target_corner_on_reset_and_step() -> None:
    env = Game2048Env(
        Config(
            target_corner="top_left",
            reward_max_tile_in_corner_bonus=0.5,
            reward_max_tile_out_of_corner_penalty=0.1,
            spawn_prob_2=1.0,
            spawn_prob_4=0.0,
        )
    )
    env.reset(seed=37)
    env.board[:, :] = np.array(
        [
            [0, 4, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    _, reward, _, _, info = env.step(LEFT)

    assert info["max_tile_in_target_corner"] is True
    assert reward == pytest.approx(env.config.reward_max_tile_in_corner_bonus)
