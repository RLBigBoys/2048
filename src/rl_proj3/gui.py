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


def run_gui(config: Config | None = None) -> None:
    """Launch the pygame window used for manual environment debugging."""
    import pygame

    env_config = config or Config()
    env = Game2048Env(env_config)
    board, info = env.reset(seed=env_config.seed)

    pygame.init()
    pygame.display.set_caption("2048 Debug GUI")
    screen = pygame.display.set_mode((env_config.window_width, env_config.window_height))
    clock = pygame.time.Clock()

    header_font = pygame.font.SysFont("arial", 28, bold=True)
    info_font = pygame.font.SysFont("arial", 22, bold=False)
    tile_font = pygame.font.SysFont("arial", 30, bold=True)

    action_by_key = {
        pygame.K_UP: UP,
        pygame.K_DOWN: DOWN,
        pygame.K_LEFT: LEFT,
        pygame.K_RIGHT: RIGHT,
    }

    last_reward = 0.0
    terminated = False
    truncated = False

    def draw_status_panel(current_info: dict[str, Any]) -> None:
        """Draw the top panel with scalar environment statistics."""
        panel_rect = pygame.Rect(
            env_config.window_padding,
            env_config.window_padding,
            env_config.window_width - (2 * env_config.window_padding),
            env_config.header_height - env_config.window_padding,
        )
        pygame.draw.rect(screen, env_config.panel_color, panel_rect, border_radius=12)

        title_surface = header_font.render("2048 RL Debugger", True, env_config.accent_text_color)
        screen.blit(
            title_surface,
            (panel_rect.x + 16, panel_rect.y + 12),
        )

        stats_text = (
            f"Score: {current_info['score']}    "
            f"Max Tile: {current_info['max_tile']}    "
            f"Reward: {last_reward:.3f}    "
            f"Step: {current_info['step_count']}"
        )
        stats_surface = info_font.render(stats_text, True, env_config.accent_text_color)
        screen.blit(
            stats_surface,
            (panel_rect.x + 16, panel_rect.y + 58),
        )

        if truncated:
            message = "Episode truncated: step limit reached. Press R to reset."
            message_color = env_config.status_fail_color
        elif terminated:
            message = "Episode terminated: no valid moves left. Press R to reset."
            message_color = env_config.status_fail_color
        elif current_info["max_tile"] >= env_config.target_tile:
            message = "Target tile reached. Continue playing or press R to reset."
            message_color = env_config.status_win_color
        else:
            message = "Controls: arrows move, R resets the board."
            message_color = env_config.accent_text_color

        message_surface = info_font.render(message, True, message_color)
        screen.blit(
            message_surface,
            (panel_rect.x + 16, panel_rect.y + 88),
        )

    def draw_board(current_board: np.ndarray) -> None:
        """Draw the board background, cells, and tile values."""
        board_x = env_config.window_padding
        board_y = env_config.header_height + env_config.window_padding
        board_rect = pygame.Rect(
            board_x,
            board_y,
            env_config.board_pixel_size,
            env_config.board_pixel_size,
        )
        pygame.draw.rect(screen, env_config.board_color, board_rect, border_radius=12)

        for row_index in range(env_config.board_size):
            for col_index in range(env_config.board_size):
                cell_x = board_x + env_config.cell_gap + (
                    col_index * (env_config.cell_size + env_config.cell_gap)
                )
                cell_y = board_y + env_config.cell_gap + (
                    row_index * (env_config.cell_size + env_config.cell_gap)
                )
                cell_rect = pygame.Rect(
                    cell_x,
                    cell_y,
                    env_config.cell_size,
                    env_config.cell_size,
                )

                tile_value = int(current_board[row_index, col_index])
                pygame.draw.rect(
                    screen,
                    _get_tile_color(env_config, tile_value),
                    cell_rect,
                    border_radius=10,
                )
                pygame.draw.rect(
                    screen,
                    env_config.grid_line_color,
                    cell_rect,
                    width=2,
                    border_radius=10,
                )

                if tile_value != 0:
                    tile_surface = tile_font.render(
                        str(tile_value),
                        True,
                        _get_tile_text_color(env_config, tile_value),
                    )
                    tile_rect = tile_surface.get_rect(center=cell_rect.center)
                    screen.blit(tile_surface, tile_rect)

    running = True
    while running:
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

        screen.fill(env_config.background_color)
        draw_status_panel(info)
        draw_board(board)
        pygame.display.flip()
        clock.tick(env_config.fps)

    pygame.quit()


if __name__ == "__main__":
    run_gui()
