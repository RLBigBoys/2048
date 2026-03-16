from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rl_proj3.config import Config


UP: int = 0
DOWN: int = 1
LEFT: int = 2
RIGHT: int = 3

ACTIONS: tuple[int, int, int, int] = (UP, DOWN, LEFT, RIGHT)

BoardArray = NDArray[np.int32]
LineArray = NDArray[np.int32]


class Game2048Env:
    """Pure NumPy implementation of the 2048 game environment."""

    _ACTION_TO_ROTATION: dict[int, int] = {
        LEFT: 0,
        UP: 1,
        RIGHT: 2,
        DOWN: 3,
    }

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the environment without starting an episode."""
        self.config: Config = config or Config()
        self._rng: np.random.Generator = np.random.default_rng(self.config.seed)
        self.board: BoardArray = np.zeros(
            (self.config.board_size, self.config.board_size),
            dtype=np.int32,
        )
        self.score: int = 0
        self.step_count: int = 0
        self._steps_without_max_tile_progress: int = 0
        self._terminated: bool = False
        self._truncated: bool = False

    def seed(self, seed: int) -> None:
        """Reset the internal random generator state."""
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> tuple[BoardArray, dict[str, Any]]:
        """Reset the board, spawn two tiles, and return the initial observation."""
        if seed is not None:
            self.seed(seed)

        self.board.fill(0)
        self.score = 0
        self.step_count = 0
        self._steps_without_max_tile_progress = 0
        self._terminated = False
        self._truncated = False

        self._spawn_tile()
        self._spawn_tile()

        info = self._build_info(
            changed=False,
            merge_gain=0,
            delta_empty=0,
            delta_max_exp=0,
            reached_target_tile=False,
            stagnation_steps=0,
            stagnation_penalty_applied=False,
            max_tile_in_target_corner=self._max_tile_in_target_corner(self.board),
        )
        return self.board.copy(), info

    def get_valid_actions(self) -> list[int]:
        """Return all actions that change the current board state."""
        return self._get_valid_actions_for_board(self.board)

    @staticmethod
    def _max_tile(board: BoardArray) -> int:
        """Return the largest tile value on the board."""
        return int(np.max(board))

    @staticmethod
    def _count_empty_cells(board: BoardArray) -> int:
        """Return the number of empty cells on the board."""
        return int(np.count_nonzero(board == 0))

    def _target_corner_index(self) -> tuple[int, int]:
        """Return the board index of the configured target corner."""
        last_index = self.config.board_size - 1
        corner_to_index = {
            "top_left": (0, 0),
            "top_right": (0, last_index),
            "bottom_left": (last_index, 0),
            "bottom_right": (last_index, last_index),
        }
        return corner_to_index[self.config.target_corner]

    def _max_tile_in_target_corner(self, board: BoardArray) -> bool:
        """Return True when a maximum tile currently occupies the target corner."""
        max_tile = self._max_tile(board)
        if max_tile <= 0:
            return False
        row_index, col_index = self._target_corner_index()
        return int(board[row_index, col_index]) == max_tile

    def _snake_indices(self) -> list[tuple[int, int]]:
        """Return board indices following a snake pattern from the target corner."""
        size = self.config.board_size
        indices: list[tuple[int, int]] = []

        if self.config.target_corner in ("top_left", "top_right"):
            row_range = range(0, size)
        else:
            row_range = range(size - 1, -1, -1)

        for row_offset, row in enumerate(row_range):
            if (
                self.config.target_corner in ("top_left", "bottom_left")
                and row_offset % 2 == 0
            ) or (
                self.config.target_corner in ("top_right", "bottom_right")
                and row_offset % 2 == 1
            ):
                col_range = range(0, size)
            else:
                col_range = range(size - 1, -1, -1)

            for col in col_range:
                indices.append((row, col))

        return indices

    def _snake_alignment_score(self, board: BoardArray) -> float:
        """Return a scalar score for how well tiles follow a snake pattern."""
        indices = self._snake_indices()
        exponents: list[int] = [
            self._tile_to_log2(int(board[row, col])) for row, col in indices
        ]
        score = 0.0
        length = len(exponents)
        for position, exponent in enumerate(exponents):
            weight = length - position
            score += weight * float(exponent)
        return score

    @staticmethod
    def _tile_to_log2(tile_value: int) -> int:
        """Convert a tile value to an integer base-2 exponent."""
        if tile_value <= 0:
            return 0
        return tile_value.bit_length() - 1

    @staticmethod
    def _build_action_mask(valid_actions: list[int]) -> NDArray[np.int8]:
        """Build a binary action mask aligned with ``ACTIONS``."""
        valid_action_set = set(valid_actions)
        return np.array(
            [int(action in valid_action_set) for action in ACTIONS],
            dtype=np.int8,
        )

    def _build_info(
        self,
        *,
        changed: bool,
        merge_gain: int,
        delta_empty: int,
        delta_max_exp: int,
        reached_target_tile: bool,
        stagnation_steps: int,
        stagnation_penalty_applied: bool,
        max_tile_in_target_corner: bool,
    ) -> dict[str, Any]:
        """Assemble the standard ``info`` dictionary for the current state."""
        valid_actions = self._get_valid_actions_for_board(self.board)
        return {
            "score": self.score,
            "max_tile": self._max_tile(self.board),
            "step_count": self.step_count,
            "valid_actions": valid_actions,
            "action_mask": self._build_action_mask(valid_actions),
            "changed": changed,
            "merge_gain": merge_gain,
            "delta_empty": delta_empty,
            "delta_max_exp": delta_max_exp,
            "reached_target_tile": reached_target_tile,
            "stagnation_steps": stagnation_steps,
            "stagnation_penalty_applied": stagnation_penalty_applied,
            "max_tile_in_target_corner": max_tile_in_target_corner,
        }

    def _spawn_tile(self) -> bool:
        """Spawn a new tile in a random empty cell and return ``True`` on success."""
        empty_positions = np.argwhere(self.board == 0)
        if empty_positions.size == 0:
            return False

        flat_index = int(self._rng.integers(0, len(empty_positions)))
        row_index, col_index = empty_positions[flat_index]
        tile_value = 2 if float(self._rng.random()) < self.config.spawn_prob_2 else 4
        self.board[int(row_index), int(col_index)] = np.int32(tile_value)
        return True

    @staticmethod
    def _compress_and_merge_line(line: LineArray) -> tuple[LineArray, int]:
        """Slide a 1D line to the left and merge equal neighbors once per move."""
        if line.ndim != 1:
            raise ValueError("line must be one-dimensional.")

        non_zero_values = [int(value) for value in line if int(value) != 0]
        merged_values: list[int] = []
        merge_gain = 0
        value_index = 0

        while value_index < len(non_zero_values):
            current_value = non_zero_values[value_index]
            next_index = value_index + 1

            if next_index < len(non_zero_values) and non_zero_values[next_index] == current_value:
                merged_tile = current_value * 2
                merged_values.append(merged_tile)
                merge_gain += merged_tile
                value_index += 2
                continue

            merged_values.append(current_value)
            value_index += 1

        merged_values.extend([0] * (line.shape[0] - len(merged_values)))
        return np.array(merged_values, dtype=np.int32), merge_gain

    def _apply_move_on_board(
        self,
        board: BoardArray,
        action: int,
    ) -> tuple[bool, BoardArray, int]:
        """Apply an action to a board copy using the rotation-based implementation."""
        if action not in self._ACTION_TO_ROTATION:
            raise ValueError(f"Unknown action: {action}.")
        if board.shape != (self.config.board_size, self.config.board_size):
            raise ValueError(
                f"board must have shape {(self.config.board_size, self.config.board_size)}.",
            )

        rotation_count = self._ACTION_TO_ROTATION[action]
        rotated_board = np.rot90(board, k=rotation_count)
        rotated_result = np.zeros_like(rotated_board)
        total_merge_gain = 0

        for row_index in range(self.config.board_size):
            merged_line, merge_gain = self._compress_and_merge_line(rotated_board[row_index, :])
            rotated_result[row_index, :] = merged_line
            total_merge_gain += merge_gain

        restored_board = np.rot90(rotated_result, k=-rotation_count).astype(np.int32, copy=False)
        changed = not np.array_equal(board, restored_board)
        return changed, restored_board.copy(), total_merge_gain

    def _get_valid_actions_for_board(self, board: BoardArray) -> list[int]:
        """Return valid actions for an arbitrary board."""
        valid_actions: list[int] = []
        for action in ACTIONS:
            changed, _, _ = self._apply_move_on_board(board, action)
            if changed:
                valid_actions.append(action)
        return valid_actions

    def step(self, action: int) -> tuple[BoardArray, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one action."""
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}.")
        if self._terminated or self._truncated:
            raise RuntimeError("Episode has finished. Call reset() before the next step().")

        board_before = self.board.copy()
        empty_before = self._count_empty_cells(board_before)
        max_tile_before = self._max_tile(board_before)
        max_exp_before = self._tile_to_log2(max_tile_before)

        self.step_count += 1

        changed, moved_board, merge_gain = self._apply_move_on_board(board_before, action)

        if not changed:
            reward = float(self.config.invalid_move_penalty)
            delta_empty = 0
            delta_max_exp = 0
            reached_target_tile = False
            self._steps_without_max_tile_progress += 1
            stagnation_penalty_applied = False
            if (
                self.config.stagnation_penalty_after_steps > 0
                and self._steps_without_max_tile_progress >= self.config.stagnation_penalty_after_steps
            ):
                reward -= self.config.reward_stagnation_penalty
                stagnation_penalty_applied = True
            max_tile_in_target_corner = self._max_tile_in_target_corner(self.board)
        else:
            empty_after_move = self._count_empty_cells(moved_board)
            max_tile_after_move = self._max_tile(moved_board)
            max_exp_after_move = self._tile_to_log2(max_tile_after_move)
            reached_target_tile = max_tile_after_move >= self.config.target_tile
            max_tile_in_target_corner = self._max_tile_in_target_corner(moved_board)

            delta_empty = empty_after_move - empty_before
            delta_max_exp = max_exp_after_move - max_exp_before

            snake_score_before = self._snake_alignment_score(board_before)
            snake_score_after = self._snake_alignment_score(moved_board)
            delta_snake = snake_score_after - snake_score_before

            large_merge_bonus = (
                self.config.reward_large_merge_factor * float(merge_gain ** 2)
                if merge_gain > 0
                else 0.0
            )

            reward = (
                (merge_gain * self.config.reward_score_scale)
                + large_merge_bonus
                + (delta_empty * self.config.reward_empty_bonus)
                + (delta_max_exp * self.config.reward_max_tile_bonus)
                + (self.config.reward_snake_factor * float(delta_snake))
            )
            if delta_max_exp > 0:
                self._steps_without_max_tile_progress = 0
            else:
                self._steps_without_max_tile_progress += 1
            stagnation_penalty_applied = False
            if (
                self.config.stagnation_penalty_after_steps > 0
                and self._steps_without_max_tile_progress >= self.config.stagnation_penalty_after_steps
            ):
                reward -= self.config.reward_stagnation_penalty
                stagnation_penalty_applied = True
            if max_tile_in_target_corner:
                reward += self.config.reward_max_tile_in_corner_bonus
            else:
                reward -= self.config.reward_max_tile_out_of_corner_penalty
            if reached_target_tile:
                reward += self.config.reward_target_tile_bonus

            self.board[:, :] = moved_board
            self.score += merge_gain
            self._spawn_tile()

        valid_actions = self.get_valid_actions()
        terminated = reached_target_tile and self.config.terminate_on_target_tile
        if not terminated:
            terminated = len(valid_actions) == 0
        truncated = self.step_count >= self.config.max_steps_per_episode

        self._terminated = terminated
        self._truncated = truncated

        info = self._build_info(
            changed=changed,
            merge_gain=merge_gain if changed else 0,
            delta_empty=delta_empty,
            delta_max_exp=delta_max_exp,
            reached_target_tile=reached_target_tile,
            stagnation_steps=self._steps_without_max_tile_progress,
            stagnation_penalty_applied=stagnation_penalty_applied,
            max_tile_in_target_corner=max_tile_in_target_corner,
        )
        return self.board.copy(), float(reward), terminated, truncated, info
