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
) -> str:
    """Build a human-readable status line for the debug panel."""
    valid_actions_text = _format_valid_actions(current_info.get("valid_actions", []))

    if truncated:
        return "Status: step limit reached | press R to reset"
    if terminated:
        return "Status: no valid moves left | press R to reset"
    if current_info["max_tile"] >= target_tile:
        return "Status: target tile reached | continue or press R"
    return f"Controls: arrows move | R reset | Valid actions: {valid_actions_text}"


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

    header_font = pygame.font.SysFont("arial", 24, bold=True)
    info_font = pygame.font.SysFont("arial", 18, bold=False)
    detail_font = pygame.font.SysFont("arial", 15, bold=False)
    tile_font = pygame.font.SysFont("arial", 28, bold=True)

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
        panel_text_color = env_config.debug_text_color
        panel_rect = pygame.Rect(
            env_config.window_padding,
            env_config.window_padding,
            env_config.window_width - (2 * env_config.window_padding),
            env_config.header_height,
        )
        pygame.draw.rect(screen, env_config.panel_color, panel_rect, border_radius=12)

        title_surface = header_font.render("2048 RL Debugger", True, panel_text_color)
        screen.blit(
            title_surface,
            (panel_rect.x + 16, panel_rect.y + 12),
        )

        left_column_x = panel_rect.x + 16
        right_column_x = panel_rect.centerx + 8
        first_metrics_y = panel_rect.y + 50
        second_metrics_y = panel_rect.y + 74

        score_surface = info_font.render(
            f"Score: {current_info['score']}",
            True,
            panel_text_color,
        )
        max_tile_surface = info_font.render(
            f"Max Tile: {current_info['max_tile']}",
            True,
            panel_text_color,
        )
        reward_surface = info_font.render(
            f"Reward: {last_reward:.3f}",
            True,
            panel_text_color,
        )
        step_surface = info_font.render(
            f"Step: {current_info['step_count']}",
            True,
            panel_text_color,
        )

        screen.blit(score_surface, (left_column_x, first_metrics_y))
        screen.blit(max_tile_surface, (left_column_x, second_metrics_y))
        screen.blit(reward_surface, (right_column_x, first_metrics_y))
        screen.blit(step_surface, (right_column_x, second_metrics_y))

        message = _format_status_text(
            current_info,
            truncated=truncated,
            terminated=terminated,
            target_tile=env_config.target_tile,
        )
        message_surface = detail_font.render(message, True, panel_text_color)
        screen.blit(
            message_surface,
            (panel_rect.x + 16, panel_rect.y + 106),
        )

        debug_text = (
            f"Board changed: {current_info['changed']} | Merge gain: {current_info['merge_gain']} | "
            f"Empty delta: {current_info['delta_empty']} | Max exp delta: {current_info['delta_max_exp']}"
        )
        debug_surface = detail_font.render(debug_text, True, panel_text_color)
        screen.blit(
            debug_surface,
            (panel_rect.x + 16, panel_rect.y + 128),
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
