"""Hexagonal Tic-Tac-Toe game logic.

Rules:
- Played on an infinite hex grid.
- Player A places 1 stone first, then players alternate placing 2 stones each.
- First player to get win_length in a row along any hex axis wins.
- Uses axial coordinates (q, r) with implicit s = -q - r.

Includes ToroidalHexGame: same rules on a fixed 25x25 toroidal grid where
coordinates wrap. This bounds computation for MCTS while approximating the
infinite grid (games rarely spread beyond 25 cells).
"""

from enum import Enum
from dataclasses import dataclass, field


class Player(Enum):
    NONE = 0
    A = 1
    B = 2


# The three line directions in axial coordinates
HEX_DIRECTIONS = [(1, 0), (0, 1), (1, -1)]


@dataclass
class HexGame:
    win_length: int = 6

    board: dict = field(default_factory=dict, init=False, repr=False)
    current_player: Player = field(default=Player.A, init=False)
    moves_left_in_turn: int = field(default=1, init=False)
    move_count: int = field(default=0, init=False)
    winner: Player = field(default=Player.NONE, init=False)
    winning_cells: list = field(default_factory=list, init=False)
    game_over: bool = field(default=False, init=False)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.board = {}  # sparse: only occupied cells
        self.current_player = Player.A
        self.moves_left_in_turn = 1  # A gets only 1 move on the first turn
        self.move_count = 0
        self.winner = Player.NONE
        self.winning_cells = []
        self.game_over = False

    def is_valid_move(self, q, r):
        if self.game_over:
            return False
        return (q, r) not in self.board

    def save_state(self):
        """Snapshot the mutable state (for undo in search)."""
        return (
            self.current_player,
            self.moves_left_in_turn,
            self.winner,
            self.winning_cells[:],
            self.game_over,
        )

    def undo_move(self, q, r, state):
        """Undo a move and restore the saved state."""
        del self.board[(q, r)]
        self.move_count -= 1
        (self.current_player, self.moves_left_in_turn,
         self.winner, self.winning_cells, self.game_over) = state

    def make_move(self, q, r):
        """Place a stone at (q, r). Returns True if the move was valid."""
        if not self.is_valid_move(q, r):
            return False

        self.board[(q, r)] = self.current_player
        self.move_count += 1

        if self._check_win(q, r):
            self.winner = self.current_player
            self.game_over = True
            return True

        self.moves_left_in_turn -= 1
        if self.moves_left_in_turn <= 0:
            self._switch_player()

        return True

    def _switch_player(self):
        if self.current_player == Player.A:
            self.current_player = Player.B
        else:
            self.current_player = Player.A
        self.moves_left_in_turn = 2

    def _check_win(self, q, r):
        """Check if placing at (q, r) creates win_length in a row."""
        player = self.board[(q, r)]

        for dq, dr in HEX_DIRECTIONS:
            cells = [(q, r)]
            for i in range(1, self.win_length):
                nq, nr = q + dq * i, r + dr * i
                if self.board.get((nq, nr)) == player:
                    cells.append((nq, nr))
                else:
                    break
            for i in range(1, self.win_length):
                nq, nr = q - dq * i, r - dr * i
                if self.board.get((nq, nr)) == player:
                    cells.append((nq, nr))
                else:
                    break
            if len(cells) >= self.win_length:
                self.winning_cells = cells
                return True

        return False


# ── Toroidal variant ─────────────────────────────────────────────────

TORUS_SIZE = 25


class ToroidalHexGame:
    """Same rules as HexGame on a fixed 25x25 toroidal grid (wrapping coords).

    Duck-typed to match HexGame's interface so MCTS code can use either.
    """

    __slots__ = (
        "win_length", "board", "current_player", "moves_left_in_turn",
        "move_count", "winner", "game_over",
    )

    def __init__(self, win_length: int = 6):
        self.win_length = win_length
        self.board: dict[tuple[int, int], Player] = {}
        self.current_player: Player = Player.A
        self.moves_left_in_turn: int = 1
        self.move_count: int = 0
        self.winner: Player = Player.NONE
        self.game_over: bool = False

    def reset(self):
        self.board = {}
        self.current_player = Player.A
        self.moves_left_in_turn = 1
        self.move_count = 0
        self.winner = Player.NONE
        self.game_over = False

    def is_valid_move(self, q: int, r: int) -> bool:
        if self.game_over:
            return False
        return (q % TORUS_SIZE, r % TORUS_SIZE) not in self.board

    def save_state(self):
        return (
            self.current_player,
            self.moves_left_in_turn,
            self.winner,
            self.game_over,
        )

    def undo_move(self, q: int, r: int, state):
        wq, wr = q % TORUS_SIZE, r % TORUS_SIZE
        del self.board[(wq, wr)]
        self.move_count -= 1
        (self.current_player, self.moves_left_in_turn,
         self.winner, self.game_over) = state

    def make_move(self, q: int, r: int) -> bool:
        wq, wr = q % TORUS_SIZE, r % TORUS_SIZE
        if self.game_over or (wq, wr) in self.board:
            return False

        self.board[(wq, wr)] = self.current_player
        self.move_count += 1

        if self._check_win(wq, wr):
            self.winner = self.current_player
            self.game_over = True
            return True

        self.moves_left_in_turn -= 1
        if self.moves_left_in_turn <= 0:
            self._switch_player()

        return True

    def _switch_player(self):
        if self.current_player == Player.A:
            self.current_player = Player.B
        else:
            self.current_player = Player.A
        self.moves_left_in_turn = 2

    def _check_win(self, q: int, r: int) -> bool:
        player = self.board[(q, r)]
        N = TORUS_SIZE

        for dq, dr in HEX_DIRECTIONS:
            count = 1
            for i in range(1, self.win_length):
                nq = (q + dq * i) % N
                nr = (r + dr * i) % N
                if self.board.get((nq, nr)) == player:
                    count += 1
                else:
                    break
            for i in range(1, self.win_length):
                nq = (q - dq * i) % N
                nr = (r - dr * i) % N
                if self.board.get((nq, nr)) == player:
                    count += 1
                else:
                    break
            if count >= self.win_length:
                return True

        return False

    def to_dict(self) -> dict:
        """Serialize game state to a JSON-compatible dict."""
        return {
            "board": {f"{q},{r}": p.value for (q, r), p in self.board.items()},
            "current_player": self.current_player.value,
            "moves_left_in_turn": self.moves_left_in_turn,
            "move_count": self.move_count,
            "winner": self.winner.value,
            "game_over": self.game_over,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToroidalHexGame":
        """Restore a game from a serialized dict."""
        game = cls()
        game.board = {
            tuple(int(x) for x in k.split(",")): Player(v)
            for k, v in d["board"].items()
        }
        game.current_player = Player(d["current_player"])
        game.moves_left_in_turn = d["moves_left_in_turn"]
        game.move_count = d["move_count"]
        game.winner = Player(d["winner"])
        game.game_over = d["game_over"]
        return game

    @classmethod
    def from_hex_game(cls, game, anchor_q: int = None, anchor_r: int = None):
        """Create a ToroidalHexGame from a HexGame by translating coordinates.

        Maps real (q, r) -> ((q + anchor_q) % TORUS_SIZE, (r + anchor_r) % TORUS_SIZE).
        Default anchor centers the origin on the torus.
        """
        if anchor_q is None:
            anchor_q = TORUS_SIZE // 2
        if anchor_r is None:
            anchor_r = TORUS_SIZE // 2
        tg = cls(win_length=game.win_length)
        N = TORUS_SIZE
        for (q, r), player in game.board.items():
            tq = (q + anchor_q) % N
            tr = (r + anchor_r) % N
            tg.board[(tq, tr)] = player
        tg.current_player = game.current_player
        tg.moves_left_in_turn = game.moves_left_in_turn
        tg.move_count = game.move_count
        tg.winner = game.winner
        tg.game_over = game.game_over
        return tg
