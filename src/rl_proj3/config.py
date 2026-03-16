from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose
from pathlib import Path
from typing import Literal


type RGBColor = tuple[int, int, int]
type TrainingMethod = Literal["value_iteration", "policy_iteration"]
type RunMode = Literal["train", "evaluate", "gui"]
type ActionSelectionMode = Literal["weighted_majority_vote", "majority_vote"]
type CornerName = Literal["top_left", "top_right", "bottom_left", "bottom_right"]
type BestCheckpointMetric = Literal["max_tile", "score", "reward"]


@dataclass(slots=True)
class Config:
    """Configuration container for the 2048 environment and debug GUI."""

    board_size: int = 4
    target_tile: int = 2048

    max_steps_per_episode: int = 2000
    seed: int = 42

    spawn_prob_2: float = 0.9
    spawn_prob_4: float = 0.1

    invalid_move_penalty: float = -1.0
    reward_score_scale: float = 1.0 / 1024.0
    reward_empty_bonus: float = 0.02
    reward_max_tile_bonus: float = 3.0
    reward_target_tile_bonus: float = 500.0
    reward_stagnation_penalty: float = 0.01
    reward_max_tile_in_corner_bonus: float = 0.5
    reward_max_tile_out_of_corner_penalty: float = 0.1
    stagnation_penalty_after_steps: int = 20
    terminate_on_target_tile: bool = True
    target_corner: CornerName = "top_left"

    clip_exp: int = 11
    run_mode: RunMode = "train"
    training_method: TrainingMethod = "value_iteration"
    
    num_training_episodes: int = 100000

    num_policy_rounds: int = 50
    episodes_per_policy_round: int = 10
    num_evaluation_episodes: int = 10
    train_alpha: float = 0.1
    train_gamma: float = 0.99
    train_epsilon: float = 0.20
    train_epsilon_end: float = 0.01
    use_epsilon_decay: bool = True
    train_policy_tau: float = 0.3
    action_selection_mode: ActionSelectionMode = "weighted_majority_vote"
    vote_weight_max_tile: float = 1.0
    vote_weight_sum_tiles: float = 0.15
    learning_curve_dir: Path = Path("learning_curves")
    artifact_dir: Path = Path("artifacts")
    model_path: Path = Path("artifacts/agent_policy.pkl")
    resume_training: bool = True
    resume_model_path = Path("artifacts/agent_policy.pkl")
    save_checkpoints: bool = True
    checkpoint_every_n_episodes: int = 5000
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    save_best_checkpoint: bool = True
    best_checkpoint_metric: BestCheckpointMetric = "max_tile"
    best_model_path: Path = Path("artifacts/best_agent_policy.pkl")
    visualize_training: bool = False
    visualize_training_every_n_episodes: int = 1
    visualize_evaluation: bool = True
    visualization_step_delay_ms: int = 120

    header_height: int = 150
    cell_size: int = 90
    cell_gap: int = 10
    window_padding: int = 16
    fps: int = 60

    background_color: RGBColor = (250, 248, 239)
    panel_color: RGBColor = (187, 173, 160)
    board_color: RGBColor = (187, 173, 160)
    empty_cell_color: RGBColor = (205, 193, 180)
    grid_line_color: RGBColor = (173, 160, 148)
    primary_text_color: RGBColor = (119, 110, 101)
    secondary_text_color: RGBColor = (249, 246, 242)
    accent_text_color: RGBColor = (255, 255, 255)
    debug_text_color: RGBColor = (24, 24, 24)
    status_win_color: RGBColor = (111, 168, 94)
    status_fail_color: RGBColor = (192, 80, 77)

    tile_colors: dict[int, RGBColor] = field(
        default_factory=lambda: {
            0: (205, 193, 180),
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (245, 149, 99),
            32: (246, 124, 95),
            64: (246, 94, 59),
            128: (237, 207, 114),
            256: (237, 204, 97),
            512: (237, 200, 80),
            1024: (237, 197, 63),
            2048: (237, 194, 46),
        }
    )

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        self.validate()

    @property
    def board_pixel_size(self) -> int:
        """Return the pixel size of the square board area."""
        return (
            (self.board_size * self.cell_size)
            + ((self.board_size + 1) * self.cell_gap)
        )

    @property
    def window_width(self) -> int:
        """Return the GUI window width in pixels."""
        return self.board_pixel_size + (2 * self.window_padding)

    @property
    def window_height(self) -> int:
        """Return the GUI window height in pixels."""
        return self.header_height + self.board_pixel_size + (2 * self.window_padding)

    def validate(self) -> None:
        """Validate configuration values and raise ``ValueError`` on mismatch."""
        if self.board_size != 4:
            raise ValueError("Only a 4x4 board is supported in this project version.")
        if self.target_tile <= 0 or self.target_tile & (self.target_tile - 1) != 0:
            raise ValueError("target_tile must be a positive power of two.")
        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.run_mode not in ("train", "evaluate", "gui"):
            raise ValueError("run_mode must be 'train', 'evaluate', or 'gui'.")
        if self.training_method not in ("value_iteration", "policy_iteration"):
            raise ValueError("training_method must be 'value_iteration' or 'policy_iteration'.")
        if self.num_training_episodes <= 0:
            raise ValueError("num_training_episodes must be positive.")
        if self.num_policy_rounds <= 0:
            raise ValueError("num_policy_rounds must be positive.")
        if self.episodes_per_policy_round <= 0:
            raise ValueError("episodes_per_policy_round must be positive.")
        if self.num_evaluation_episodes <= 0:
            raise ValueError("num_evaluation_episodes must be positive.")
        if not 0.0 < self.train_alpha <= 1.0:
            raise ValueError("train_alpha must be in (0, 1].")
        if not 0.0 <= self.train_gamma <= 1.0:
            raise ValueError("train_gamma must be in [0, 1].")
        if not 0.0 <= self.train_epsilon <= 1.0:
            raise ValueError("train_epsilon must be in [0, 1].")
        if not 0.0 <= self.train_epsilon_end <= 1.0:
            raise ValueError("train_epsilon_end must be in [0, 1].")
        if self.train_epsilon_end > self.train_epsilon:
            raise ValueError("train_epsilon_end must be less than or equal to train_epsilon.")
        if not 0.0 < self.train_policy_tau <= 1.0:
            raise ValueError("train_policy_tau must be in (0, 1].")
        if self.action_selection_mode not in ("weighted_majority_vote", "majority_vote"):
            raise ValueError(
                "action_selection_mode must be 'weighted_majority_vote' or 'majority_vote'.",
            )
        if self.vote_weight_max_tile < 0.0:
            raise ValueError("vote_weight_max_tile must be non-negative.")
        if self.vote_weight_sum_tiles < 0.0:
            raise ValueError("vote_weight_sum_tiles must be non-negative.")
        if self.checkpoint_every_n_episodes <= 0:
            raise ValueError("checkpoint_every_n_episodes must be positive.")
        if self.best_checkpoint_metric not in ("max_tile", "score", "reward"):
            raise ValueError("best_checkpoint_metric must be 'max_tile', 'score', or 'reward'.")
        if self.visualize_training_every_n_episodes <= 0:
            raise ValueError("visualize_training_every_n_episodes must be positive.")
        if self.visualization_step_delay_ms < 0:
            raise ValueError("visualization_step_delay_ms must be non-negative.")
        if self.reward_target_tile_bonus < 0.0:
            raise ValueError("reward_target_tile_bonus must be non-negative.")
        if self.reward_stagnation_penalty < 0.0:
            raise ValueError("reward_stagnation_penalty must be non-negative.")
        if self.reward_max_tile_in_corner_bonus < 0.0:
            raise ValueError("reward_max_tile_in_corner_bonus must be non-negative.")
        if self.reward_max_tile_out_of_corner_penalty < 0.0:
            raise ValueError("reward_max_tile_out_of_corner_penalty must be non-negative.")
        if self.stagnation_penalty_after_steps < 0:
            raise ValueError("stagnation_penalty_after_steps must be non-negative.")
        if self.target_corner not in ("top_left", "top_right", "bottom_left", "bottom_right"):
            raise ValueError(
                "target_corner must be 'top_left', 'top_right', 'bottom_left', or 'bottom_right'.",
            )
        if self.spawn_prob_2 < 0.0 or self.spawn_prob_4 < 0.0:
            raise ValueError("Spawn probabilities must be non-negative.")
        if not isclose(self.spawn_prob_2 + self.spawn_prob_4, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("spawn_prob_2 + spawn_prob_4 must be equal to 1.0.")
        if self.clip_exp < 0:
            raise ValueError("clip_exp must be non-negative.")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive.")
        if self.cell_gap < 0 or self.window_padding < 0 or self.header_height < 0:
            raise ValueError("GUI sizes must be non-negative.")
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
