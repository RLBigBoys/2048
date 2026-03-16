from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from rl_proj3.config import Config
from rl_proj3.env import DOWN, LEFT, RIGHT, UP, Game2048Env


def _get_tile_color(config: Config, value: int) -> tuple[int, int, int]:
    """Return a display color for a tile value."""
    return config.tile_colors.get(value, (60, 58, 50))


def _get_tile_text_color(config: Config, value: int) -> tuple[int, int, int]:
    """Return a text color with enough contrast for the tile value."""
    if value <= 4:
        return config.primary_text_color
    return config.secondary_text_color


def _format_valid_actions(valid_actions: list[int]) -> str:
    """Return a compact, stable string with valid action names."""
    action_names: list[str] = []
    for action, label in ((UP, "U"), (DOWN, "D"), (LEFT, "L"), (RIGHT, "R")):
        if action in valid_actions:
            action_names.append(label)
    return ", ".join(action_names) if action_names else "-"


def _format_status_text(
    current_info: dict[str, Any],
    *,
    truncated: bool,
    terminated: bool,
    target_tile: int,
    mode_label: str,
) -> str:
    """Build a human-readable status line for the debug panel."""
    valid_actions_text = _format_valid_actions(current_info.get("valid_actions", []))

    if truncated:
        return f"{mode_label}: step limit reached"
    if terminated:
        return f"{mode_label}: no valid moves left"
    if current_info["max_tile"] >= target_tile:
        return f"{mode_label}: target tile reached"
    return f"{mode_label}: valid actions {valid_actions_text}"


class Pygame2048Viewer:
    """Reusable pygame renderer for manual play, training, and evaluation."""

    def __init__(self, config: Config | None = None, *, title: str = "2048 RL Debugger") -> None:
        """Initialize the pygame window and static drawing resources."""
        import pygame

        self._pygame = pygame
        self.config = config or Config()
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((self.config.window_width, self.config.window_height))
        self.clock = pygame.time.Clock()
        self.header_font = pygame.font.SysFont("arial", 24, bold=True)
        self.info_font = pygame.font.SysFont("arial", 18, bold=False)
        self.detail_font = pygame.font.SysFont("arial", 15, bold=False)
        self.tile_font = pygame.font.SysFont("arial", 28, bold=True)
        self.closed = False

    def close(self) -> None:
        """Close the pygame window exactly once."""
        if self.closed:
            return
        self._pygame.quit()
        self.closed = True

    def _handle_window_events(self) -> bool:
        """Poll basic window events and return False if the user asked to quit."""
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self.close()
                return False
            if event.type == self._pygame.KEYDOWN and event.key in (
                self._pygame.K_ESCAPE,
                self._pygame.K_q,
            ):
                self.close()
                return False
        return True

    def render(
        self,
        board: np.ndarray,
        info: dict[str, Any],
        *,
        last_reward: float,
        terminated: bool,
        truncated: bool,
        mode_label: str,
        episode_index: int | None = None,
        total_episodes: int | None = None,
        action_label: str | None = None,
    ) -> bool:
        """Draw a full frame and return False if the window was closed."""
        if self.closed:
            return False
        if not self._handle_window_events():
            return False

        self.screen.fill(self.config.background_color)
        self._draw_status_panel(
            info,
            last_reward=last_reward,
            terminated=terminated,
            truncated=truncated,
            mode_label=mode_label,
            episode_index=episode_index,
            total_episodes=total_episodes,
            action_label=action_label,
        )
        self._draw_board(board)
        self._pygame.display.flip()
        self.clock.tick(self.config.fps)
        return True

    def wait(self, delay_ms: int) -> bool:
        """Delay between frames while keeping the window responsive."""
        if self.closed:
            return False
        if not self._handle_window_events():
            return False
        if delay_ms > 0:
            self._pygame.time.wait(delay_ms)
        return True

    def _draw_status_panel(
        self,
        current_info: dict[str, Any],
        *,
        last_reward: float,
        terminated: bool,
        truncated: bool,
        mode_label: str,
        episode_index: int | None,
        total_episodes: int | None,
        action_label: str | None,
    ) -> None:
        """Draw the top panel with scalar environment statistics."""
        panel_text_color = self.config.debug_text_color
        panel_rect = self._pygame.Rect(
            self.config.window_padding,
            self.config.window_padding,
            self.config.window_width - (2 * self.config.window_padding),
            self.config.header_height,
        )
        self._pygame.draw.rect(self.screen, self.config.panel_color, panel_rect, border_radius=12)

        title_surface = self.header_font.render("2048 RL Debugger", True, panel_text_color)
        self.screen.blit(title_surface, (panel_rect.x + 16, panel_rect.y + 12))

        left_column_x = panel_rect.x + 16
        right_column_x = panel_rect.centerx + 8
        first_metrics_y = panel_rect.y + 50
        second_metrics_y = panel_rect.y + 74

        stats = (
            (f"Score: {current_info['score']}", left_column_x, first_metrics_y),
            (f"Max Tile: {current_info['max_tile']}", left_column_x, second_metrics_y),
            (f"Reward: {last_reward:.3f}", right_column_x, first_metrics_y),
            (f"Step: {current_info['step_count']}", right_column_x, second_metrics_y),
        )
        for text, x_pos, y_pos in stats:
            surface = self.info_font.render(text, True, panel_text_color)
            self.screen.blit(surface, (x_pos, y_pos))

        message = _format_status_text(
            current_info,
            truncated=truncated,
            terminated=terminated,
            target_tile=self.config.target_tile,
            mode_label=mode_label,
        )
        self.screen.blit(
            self.detail_font.render(message, True, panel_text_color),
            (panel_rect.x + 16, panel_rect.y + 106),
        )

        progress_parts: list[str] = []
        if episode_index is not None and total_episodes is not None:
            progress_parts.append(f"Episode {episode_index}/{total_episodes}")
        if action_label is not None:
            progress_parts.append(f"Action: {action_label}")
        progress_text = " | ".join(progress_parts) if progress_parts else "Press Q or Esc to close"
        self.screen.blit(
            self.detail_font.render(progress_text, True, panel_text_color),
            (panel_rect.x + 16, panel_rect.y + 128),
        )

    def _draw_board(self, current_board: np.ndarray) -> None:
        """Draw the board background, cells, and tile values."""
        board_x = self.config.window_padding
        board_y = self.config.header_height + self.config.window_padding
        board_rect = self._pygame.Rect(
            board_x,
            board_y,
            self.config.board_pixel_size,
            self.config.board_pixel_size,
        )
        self._pygame.draw.rect(self.screen, self.config.board_color, board_rect, border_radius=12)

        for row_index in range(self.config.board_size):
            for col_index in range(self.config.board_size):
                cell_x = board_x + self.config.cell_gap + (
                    col_index * (self.config.cell_size + self.config.cell_gap)
                )
                cell_y = board_y + self.config.cell_gap + (
                    row_index * (self.config.cell_size + self.config.cell_gap)
                )
                cell_rect = self._pygame.Rect(
                    cell_x,
                    cell_y,
                    self.config.cell_size,
                    self.config.cell_size,
                )

                tile_value = int(current_board[row_index, col_index])
                self._pygame.draw.rect(
                    self.screen,
                    _get_tile_color(self.config, tile_value),
                    cell_rect,
                    border_radius=10,
                )
                self._pygame.draw.rect(
                    self.screen,
                    self.config.grid_line_color,
                    cell_rect,
                    width=2,
                    border_radius=10,
                )

                if tile_value != 0:
                    tile_surface = self.tile_font.render(
                        str(tile_value),
                        True,
                        _get_tile_text_color(self.config, tile_value),
                    )
                    tile_rect = tile_surface.get_rect(center=cell_rect.center)
                    self.screen.blit(tile_surface, tile_rect)


def run_gui(config: Config | None = None) -> None:
    """Launch the pygame window used for manual environment debugging."""
    import pygame

    env_config = config or Config()
    env = Game2048Env(env_config)
    board, info = env.reset(seed=env_config.seed)
    viewer = Pygame2048Viewer(env_config)

    action_by_key = {
        pygame.K_UP: UP,
        pygame.K_DOWN: DOWN,
        pygame.K_LEFT: LEFT,
        pygame.K_RIGHT: RIGHT,
    }

    last_reward = 0.0
    terminated = False
    truncated = False
    running = True

    while running and not viewer.closed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if event.type != pygame.KEYDOWN:
                continue

            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
                continue

            if event.key == pygame.K_r:
                board, info = env.reset(seed=env_config.seed)
                last_reward = 0.0
                terminated = False
                truncated = False
                continue

            if event.key in action_by_key and not (terminated or truncated):
                board, last_reward, terminated, truncated, info = env.step(action_by_key[event.key])

        if not running:
            break

        if not viewer.render(
            board,
            info,
            last_reward=last_reward,
            terminated=terminated,
            truncated=truncated,
            mode_label="Manual",
        ):
            break

    viewer.close()


if __name__ == "__main__":
    run_gui()
