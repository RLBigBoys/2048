from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_proj3.features import (
    BLOCKS_2X2,
    BLOCKS_2X3,
    BLOCKS_3X2,
    BLOCK_SLICES_2X2,
    BLOCK_SLICES_2X3,
    BLOCK_SLICES_3X2,
    BlockExtractor,
    BlockExtractor2x3,
    BlockExtractor3x2,
    ColExtractor,
    RowExtractor,
    tile_to_exp,
)


def test_tile_to_exp_applies_clipping() -> None:
    assert tile_to_exp(0, clip_exp=8) == 0
    assert tile_to_exp(2, clip_exp=8) == 1
    assert tile_to_exp(4, clip_exp=8) == 2
    assert tile_to_exp(8, clip_exp=8) == 3
    assert tile_to_exp(256, clip_exp=8) == 8
    assert tile_to_exp(512, clip_exp=8) == 8
    assert tile_to_exp(1024, clip_exp=8) == 8


def test_block_extractor_returns_expected_slices() -> None:
    board = np.array(
        [
            [0, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2048],
            [2, 4, 8, 16],
        ],
        dtype=np.int32,
    )

    features = BlockExtractor(clip_exp=8).extract(board)

    assert len(BLOCK_SLICES_2X2) == 9
    assert BLOCKS_2X2 == BLOCK_SLICES_2X2
    assert len(features) == 9
    assert features[0] == (0, 1, 4, 5)
    assert features[1] == (1, 2, 5, 6)
    assert features[4] == (5, 6, 8, 8)
    assert features[-1] == (8, 8, 3, 4)


def test_row_extractor_returns_expected_rows() -> None:
    board = np.array(
        [
            [0, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2048],
            [2, 4, 8, 16],
        ],
        dtype=np.int32,
    )

    features = RowExtractor(clip_exp=8).extract(board)

    assert len(features) == 4
    assert features[0] == (0, 1, 2, 3)
    assert features[1] == (4, 5, 6, 7)
    assert features[2] == (8, 8, 8, 8)
    assert features[3] == (1, 2, 3, 4)


def test_block_extractor_2x3_returns_expected_slices() -> None:
    board = np.array(
        [
            [0, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2048],
            [2, 4, 8, 16],
        ],
        dtype=np.int32,
    )

    features = BlockExtractor2x3(clip_exp=8).extract(board)

    assert len(BLOCK_SLICES_2X3) == 6
    assert BLOCKS_2X3 == BLOCK_SLICES_2X3
    assert len(features) == 6
    assert features[0] == (0, 1, 2, 4, 5, 6)
    assert features[1] == (1, 2, 3, 5, 6, 7)
    assert features[-1] == (8, 8, 8, 2, 3, 4)


def test_block_extractor_3x2_returns_expected_slices() -> None:
    board = np.array(
        [
            [0, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2048],
            [2, 4, 8, 16],
        ],
        dtype=np.int32,
    )

    features = BlockExtractor3x2(clip_exp=8).extract(board)

    assert len(BLOCK_SLICES_3X2) == 6
    assert BLOCKS_3X2 == BLOCK_SLICES_3X2
    assert len(features) == 6
    assert features[0] == (0, 1, 4, 5, 8, 8)
    assert features[1] == (1, 2, 5, 6, 8, 8)
    assert features[-1] == (6, 7, 8, 8, 3, 4)


def test_col_extractor_returns_expected_columns() -> None:
    board = np.array(
        [
            [0, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2048],
            [2, 4, 8, 16],
        ],
        dtype=np.int32,
    )

    features = ColExtractor(clip_exp=8).extract(board)

    assert len(features) == 4
    assert features[0] == (0, 4, 8, 1)
    assert features[1] == (1, 5, 8, 2)
    assert features[2] == (2, 6, 8, 3)
    assert features[3] == (3, 7, 8, 4)
