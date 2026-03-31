"""Base class for Hex Tic-Tac-Toe bots."""

import random
from abc import ABC, abstractmethod


def hex_distance(dq, dr):
    return max(abs(dq), abs(dr), abs(dq + dr))


# Precomputed distance-2 offsets for candidate generation
_D2_OFFSETS = tuple(
    (dq, dr)
    for dq in range(-2, 3)
    for dr in range(-2, 3)
    if hex_distance(dq, dr) <= 2 and (dq, dr) != (0, 0)
)


class BoardTooLargeError(Exception):
    """Raised when the board exceeds a bot's maximum representable size."""


class Bot(ABC):
    """Abstract bot. Subclasses must implement get_move and respect time_limit."""

    pair_moves = False  # True if get_move returns both moves of a double turn

    def __init__(self, time_limit=0.05):
        self.time_limit = time_limit
        self.last_depth = 0  # depth reached on most recent get_move call

    @abstractmethod
    def get_move(self, game) -> tuple[int, int] | list[tuple[int, int]]:
        """Return (q, r) or [(q1,r1), (q2,r2)] if pair_moves is True."""
        ...

    def __str__(self):
        return self.__class__.__name__


class RandomBot(Bot):
    """Places stones randomly near existing stones. Useful as a baseline."""

    def get_move(self, game):
        if not game.board:
            return (0, 0)
        candidates = set()
        for q, r in game.board:
            for dq, dr in _D2_OFFSETS:
                nb = (q + dq, r + dr)
                if nb not in game.board:
                    candidates.add(nb)
        return random.choice(list(candidates))
