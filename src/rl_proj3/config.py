from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose


type RGBColor = tuple[int, int, int]


@dataclass(slots=True)
class Config:
    """Configuration container for the 2048 environment and debug GUI."""

    board_size: int = 4
    target_tile: int = 2048

    max_steps_per_episode: int = 1000
    seed: int = 42

    spawn_prob_2: float = 0.9
    spawn_prob_4: float = 0.1

    invalid_move_penalty: float = -1.0
    reward_score_scale: float = 1.0 / 128.0
    reward_empty_bonus: float = 0.1
    reward_max_tile_bonus: float = 1.0

    clip_exp: int = 8

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
