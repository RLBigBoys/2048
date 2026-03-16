from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import numpy as np
from numpy.typing import NDArray


type BoardArray = NDArray[np.int32]
type FeatureTuple = tuple[int, ...]
type BlockSlice = tuple[tuple[int, int], tuple[int, int]]

BLOCK_SLICES_2X2: Final[tuple[BlockSlice, ...]] = (
    ((0, 2), (0, 2)),
    ((0, 2), (1, 3)),
    ((0, 2), (2, 4)),
    ((1, 3), (0, 2)),
    ((1, 3), (1, 3)),
    ((1, 3), (2, 4)),
    ((2, 4), (0, 2)),
    ((2, 4), (1, 3)),
    ((2, 4), (2, 4)),
)
BLOCKS_2X2: Final[tuple[BlockSlice, ...]] = BLOCK_SLICES_2X2
BLOCK_SLICES_2X3: Final[tuple[BlockSlice, ...]] = (
    ((0, 2), (0, 3)),
    ((0, 2), (1, 4)),
    ((1, 3), (0, 3)),
    ((1, 3), (1, 4)),
    ((2, 4), (0, 3)),
    ((2, 4), (1, 4)),
)
BLOCKS_2X3: Final[tuple[BlockSlice, ...]] = BLOCK_SLICES_2X3
BLOCK_SLICES_3X2: Final[tuple[BlockSlice, ...]] = (
    ((0, 3), (0, 2)),
    ((0, 3), (1, 3)),
    ((0, 3), (2, 4)),
    ((1, 4), (0, 2)),
    ((1, 4), (1, 3)),
    ((1, 4), (2, 4)),
)
BLOCKS_3X2: Final[tuple[BlockSlice, ...]] = BLOCK_SLICES_3X2


def tile_to_exp(x: int, clip_exp: int) -> int:
    """Encode a tile as its clipped base-2 exponent."""
    if clip_exp < 0:
        raise ValueError("clip_exp must be non-negative.")
    if x < 0:
        raise ValueError("Tile value must be non-negative.")
    if x == 0:
        return 0
    if x & (x - 1) != 0:
        raise ValueError("Tile value must be zero or a power of two.")

    exponent = x.bit_length() - 1
    return min(exponent, clip_exp)


def _validate_board(board: BoardArray) -> None:
    """Validate that the board has the expected 4x4 shape."""
    if board.shape != (4, 4):
        raise ValueError("Board must have shape (4, 4).")


class FeatureExtractor(ABC):
    """Abstract interface for all state-aggregation extractors."""

    def __init__(self, clip_exp: int) -> None:
        """Store the clipping threshold used for tile encoding."""
        self.clip_exp = clip_exp

    @abstractmethod
    def extract(self, board: BoardArray) -> list[FeatureTuple]:
        """Extract a list of encoded feature tuples from the board."""


class BlockExtractor(FeatureExtractor):
    """Extract all nine overlapping 2x2 blocks from the 4x4 board."""

    def __init__(self, clip_exp: int) -> None:
        """Initialize the extractor with a clipping threshold."""
        super().__init__(clip_exp=clip_exp)

    def extract(self, board: BoardArray) -> list[FeatureTuple]:
        """Return encoded tuples for all predefined 2x2 block slices."""
        _validate_board(board)
        encoded_blocks: list[FeatureTuple] = []

        for (row_start, row_end), (col_start, col_end) in BLOCK_SLICES_2X2:
            block = board[row_start:row_end, col_start:col_end].reshape(-1)
            encoded_blocks.append(
                tuple(tile_to_exp(int(tile), self.clip_exp) for tile in block),
            )

        return encoded_blocks


class BlockExtractor2x3(FeatureExtractor):
    """Extract all six overlapping 2x3 blocks from the 4x4 board."""

    def __init__(self, clip_exp: int) -> None:
        """Initialize the extractor with a clipping threshold."""
        super().__init__(clip_exp=clip_exp)

    def extract(self, board: BoardArray) -> list[FeatureTuple]:
        """Return encoded tuples for all predefined 2x3 block slices."""
        _validate_board(board)
        encoded_blocks: list[FeatureTuple] = []

        for (row_start, row_end), (col_start, col_end) in BLOCK_SLICES_2X3:
            block = board[row_start:row_end, col_start:col_end].reshape(-1)
            encoded_blocks.append(
                tuple(tile_to_exp(int(tile), self.clip_exp) for tile in block),
            )

        return encoded_blocks


class BlockExtractor3x2(FeatureExtractor):
    """Extract all six overlapping 3x2 blocks from the 4x4 board."""

    def __init__(self, clip_exp: int) -> None:
        """Initialize the extractor with a clipping threshold."""
        super().__init__(clip_exp=clip_exp)

    def extract(self, board: BoardArray) -> list[FeatureTuple]:
        """Return encoded tuples for all predefined 3x2 block slices."""
        _validate_board(board)
        encoded_blocks: list[FeatureTuple] = []

        for (row_start, row_end), (col_start, col_end) in BLOCK_SLICES_3X2:
            block = board[row_start:row_end, col_start:col_end].reshape(-1)
            encoded_blocks.append(
                tuple(tile_to_exp(int(tile), self.clip_exp) for tile in block),
            )

        return encoded_blocks


class RowExtractor(FeatureExtractor):
    """Extract encoded rows from the board."""

    def __init__(self, clip_exp: int) -> None:
        """Initialize the extractor with a clipping threshold."""
        super().__init__(clip_exp=clip_exp)

    def extract(self, board: BoardArray) -> list[FeatureTuple]:
        """Return four encoded row tuples."""
        _validate_board(board)
        return [
            tuple(tile_to_exp(int(tile), self.clip_exp) for tile in board[row_index, :])
            for row_index in range(board.shape[0])
        ]


class ColExtractor(FeatureExtractor):
    """Extract encoded columns from the board."""

    def __init__(self, clip_exp: int) -> None:
        """Initialize the extractor with a clipping threshold."""
        super().__init__(clip_exp=clip_exp)

    def extract(self, board: BoardArray) -> list[FeatureTuple]:
        """Return four encoded column tuples."""
        _validate_board(board)
        return [
            tuple(tile_to_exp(int(tile), self.clip_exp) for tile in board[:, col_index])
            for col_index in range(board.shape[1])
        ]
