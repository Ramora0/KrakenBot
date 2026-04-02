"""Pygame interface for Hexagonal Tic-Tac-Toe — play against KrakenBot's MCTS.

Run:
  python play.py                                    # default self-play checkpoint
  python play.py --checkpoint path/to/model.pt      # custom checkpoint
  python play.py --n-sims 400                       # stronger search

Modes:
  PLAY   — Click to place stones, AI responds. SPACE=swap, A=autoplay.
  REVIEW — Step through move history with arrow keys. P=resume play.
  EDIT   — Left/right click to place Red/Blue. E=exit edit.

Controls: N=toggle numbers, S=save, R=restart, Q=quit.
"""

import argparse
import math
import os
import pickle
import sys
import time
import pygame
from game import HexGame, ToroidalHexGame, TORUS_SIZE, Player

# --- Modes ---
MODE_PLAY = "play"
MODE_REVIEW = "review"
MODE_EDIT = "edit"

# --- Layout ---
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 800
MAX_HEX_SIZE = 28
VISIBLE_DIST = 8

# --- Colors ---
BG_COLOR = (24, 24, 32)
EMPTY_FILL = (48, 48, 58)
GRID_LINE = (72, 72, 85)
PLAYER_A_COLOR = (220, 62, 62)
PLAYER_B_COLOR = (62, 120, 220)
PLAYER_A_HOVER = (120, 40, 40)
WIN_BORDER = (255, 215, 0)
AI_LAST_MOVE = (255, 255, 255)
TEXT_COLOR = (220, 220, 230)
SUBTLE_TEXT = (130, 130, 150)
EDIT_MODE_COLOR = (255, 180, 40)
REVIEW_COLOR = (100, 220, 100)

AI_MOVE_DELAY = 300


def _hex_distance(dq, dr):
    return max(abs(dq), abs(dr), abs(dq + dr))


_VISIBLE_OFFSETS = tuple(
    (dq, dr)
    for dq in range(-VISIBLE_DIST, VISIBLE_DIST + 1)
    for dr in range(-VISIBLE_DIST, VISIBLE_DIST + 1)
    if _hex_distance(dq, dr) <= VISIBLE_DIST
)


def hex_corners(cx, cy, size):
    return [
        (cx + size * math.cos(math.radians(60 * i + 30)),
         cy + size * math.sin(math.radians(60 * i + 30)))
        for i in range(6)
    ]


def hex_to_pixel(q, r, size, ox, oy):
    x = size * math.sqrt(3) * (q + r * 0.5) + ox
    y = size * 1.5 * r + oy
    return x, y


def pixel_to_hex(mx, my, size, ox, oy):
    px = (mx - ox) / size
    py = (my - oy) / size
    r_frac = 2.0 / 3 * py
    q_frac = px / math.sqrt(3) - r_frac / 2
    s_frac = -q_frac - r_frac
    rq, rr, rs = round(q_frac), round(r_frac), round(s_frac)
    dq = abs(rq - q_frac)
    dr = abs(rr - r_frac)
    ds = abs(rs - s_frac)
    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    return int(rq), int(rr)


def get_visible_cells(game):
    board = game.board
    if not board:
        return {(oq, or_) for oq, or_ in _VISIBLE_OFFSETS}
    cells = set()
    for q, r in board:
        for oq, or_ in _VISIBLE_OFFSETS:
            cells.add((q + oq, r + or_))
    return cells


def get_visible_cells_torus():
    return {(q, r) for q in range(TORUS_SIZE) for r in range(TORUS_SIZE)}


def compute_view(visible_cells):
    if not visible_cells:
        return MAX_HEX_SIZE, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2
    S3 = math.sqrt(3)
    uxs = [S3 * (q + r * 0.5) for q, r in visible_cells]
    uys = [1.5 * r for q, r in visible_cells]
    min_ux, max_ux = min(uxs), max(uxs)
    min_uy, max_uy = min(uys), max(uys)
    ext_x = max_ux - min_ux + S3
    ext_y = max_uy - min_uy + 2
    avail_x = WINDOW_WIDTH - 60
    avail_y = WINDOW_HEIGHT - 140
    size = MAX_HEX_SIZE
    if ext_x > 0:
        size = min(size, avail_x / ext_x)
    if ext_y > 0:
        size = min(size, avail_y / ext_y)
    size = max(8.0, size)
    center_ux = (min_ux + max_ux) / 2
    center_uy = (min_uy + max_uy) / 2
    ox = WINDOW_WIDTH / 2 - center_ux * size
    oy = WINDOW_HEIGHT / 2 - center_uy * size + 20
    return size, ox, oy


def rebuild_game(move_list, base_board=None, base_player=Player.A,
                 game_cls=HexGame):
    game = game_cls(win_length=6)
    if base_board:
        game.board = dict(base_board)
        game.current_player = base_player
        game.move_count = len(base_board)
        game.moves_left_in_turn = 2 if base_board else 1
    move_numbers = {}
    turn_number = 0
    for q, r in move_list:
        new_turn = game.moves_left_in_turn == 2 or game.move_count == 0
        if new_turn:
            turn_number += 1
        game.make_move(q, r)
        move_numbers[(q, r)] = turn_number
    return game, move_numbers, turn_number


def draw_board(screen, game, visible_cells, hover_hex, hex_size, ox, oy, fonts,
               mode=MODE_PLAY, human_player=Player.A, ai_stats=None,
               last_ai_moves=(), edit_hover_btn=1,
               show_numbers=False, move_numbers=None,
               save_msg=None, autoplay=False,
               review_pos=0, review_total=0):
    font_big, font_med, font_sm = fonts
    screen.fill(BG_COLOR)

    board = game.board
    human_color = PLAYER_A_COLOR if human_player == Player.A else PLAYER_B_COLOR
    ai_color = PLAYER_B_COLOR if human_player == Player.A else PLAYER_A_COLOR

    # Hex cells
    for (q, r) in visible_cells:
        cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
        corners = hex_corners(cx, cy, hex_size)

        player = board.get((q, r))
        if player == Player.A:
            fill = PLAYER_A_COLOR
        elif player == Player.B:
            fill = PLAYER_B_COLOR
        elif hover_hex == (q, r) and not game.game_over:
            if mode == MODE_EDIT:
                if edit_hover_btn == 3:
                    fill = tuple(c // 2 for c in PLAYER_B_COLOR)
                else:
                    fill = tuple(c // 2 for c in PLAYER_A_COLOR)
            elif mode == MODE_PLAY:
                fill = PLAYER_A_HOVER
            else:
                fill = EMPTY_FILL
        else:
            fill = EMPTY_FILL

        pygame.draw.polygon(screen, fill, corners)
        pygame.draw.polygon(screen, GRID_LINE, corners, 2)

    # AI last move highlight
    for (q, r) in last_ai_moves:
        cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
        corners = hex_corners(cx, cy, hex_size)
        pygame.draw.polygon(screen, AI_LAST_MOVE, corners, 3)

    # Winning cells highlight
    for (q, r) in getattr(game, 'winning_cells', []):
        cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
        corners = hex_corners(cx, cy, hex_size)
        pygame.draw.polygon(screen, WIN_BORDER, corners, 3)

    # Move numbers overlay
    if show_numbers and move_numbers:
        num_font = pygame.font.SysFont("Arial", max(10, int(hex_size * 0.6)), bold=True)
        for (q, r), turn_num in move_numbers.items():
            if (q, r) in game.board:
                cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
                label = num_font.render(str(turn_num), True, (255, 255, 255))
                shadow = num_font.render(str(turn_num), True, (0, 0, 0))
                rect = label.get_rect(center=(cx, cy))
                screen.blit(shadow, shadow.get_rect(center=(cx + 1, cy + 1)))
                screen.blit(label, rect)

    # --- Status text ---
    if mode == MODE_EDIT:
        status = font_big.render("EDIT MODE", True, EDIT_MODE_COLOR)
        screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))
        hint = font_med.render(
            "Left click = Red  |  Right click = Blue  |  Click same to remove",
            True, SUBTLE_TEXT)
        screen.blit(hint, hint.get_rect(centerx=WINDOW_WIDTH // 2, y=55))
    elif mode == MODE_REVIEW:
        status = font_big.render(
            f"REVIEW  {review_pos}/{review_total}", True, REVIEW_COLOR)
        screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))
        color = PLAYER_A_COLOR if game.current_player == Player.A else PLAYER_B_COLOR
        whose = "Red" if game.current_player == Player.A else "Blue"
        if game.winner != Player.NONE:
            winner_name = "Red" if game.winner == Player.A else "Blue"
            hint_text = f"{winner_name} wins!"
        elif game.game_over:
            hint_text = "Draw!"
        else:
            hint_text = f"{whose} to move"
        hint = font_med.render(hint_text, True, color)
        screen.blit(hint, hint.get_rect(centerx=WINDOW_WIDTH // 2, y=55))
    elif game.winner != Player.NONE:
        name = "You win!" if game.winner == human_player else "AI wins!"
        color = human_color if game.winner == human_player else ai_color
        status = font_big.render(name, True, color)
        screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))
    elif game.game_over:
        status = font_big.render("Draw!", True, TEXT_COLOR)
        screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))
    elif autoplay:
        color = PLAYER_A_COLOR if game.current_player == Player.A else PLAYER_B_COLOR
        status = font_big.render("AI vs AI", True, color)
        screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))
    elif game.current_player != human_player:
        status = font_big.render("AI is thinking...", True, ai_color)
        screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))
    else:
        status = font_big.render("Your turn", True, human_color)
        screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))
        moves_surf = font_med.render(
            f"Moves left: {game.moves_left_in_turn}", True, SUBTLE_TEXT)
        screen.blit(moves_surf, moves_surf.get_rect(centerx=WINDOW_WIDTH // 2, y=55))

    if ai_stats:
        sims, = ai_stats
        ai_info = font_sm.render(f"AI: {sims} sims", True, SUBTLE_TEXT)
        screen.blit(ai_info, ai_info.get_rect(centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT - 50))

    # Bottom instructions per mode
    if mode == MODE_EDIT:
        instr = font_sm.render(
            "E = exit edit  |  S = save  |  R = restart  |  Q = quit", True, SUBTLE_TEXT)
    elif mode == MODE_REVIEW:
        instr = font_sm.render(
            "\u2190\u2192 = step  |  P = play from here  |  N = numbers  |  S = save  |  E = edit  |  R = restart  |  Q = quit",
            True, SUBTLE_TEXT)
    else:
        instr = font_sm.render(
            "\u2190\u2192 = review  |  SPACE = swap  |  A = autoplay  |  N = numbers  |  S = save  |  E = edit  |  R = restart  |  Q = quit",
            True, SUBTLE_TEXT)
    screen.blit(instr, instr.get_rect(centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT - 30))

    if save_msg:
        msg_surf = font_med.render(save_msg, True, (100, 255, 100))
        screen.blit(msg_surf, msg_surf.get_rect(centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT - 70))

    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(
        description="Play Hex Tic-Tac-Toe against KrakenBot's MCTS.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt). "
                             "Default: training/mcts_results/best.pt")
    parser.add_argument("--n-sims", type=int, default=200,
                        help="MCTS simulations per move (default: 200)")
    parser.add_argument("--position", type=str, default=None,
                        help="Load a saved position .pkl file")
    parser.add_argument("--moves", type=str, default=None,
                        help="Pre-load moves as 'q1,r1 q2,r2 ...'")
    parser.add_argument("--torus", action="store_true",
                        help="Play on 25x25 toroidal grid instead of infinite")
    args = parser.parse_args()

    # Resolve checkpoint path
    if args.checkpoint is None:
        selfplay_best = os.path.join("training", "mcts_results", "best.pt")
        distill_best = os.path.join("training", "resnet_results", "best.pt")
        if os.path.exists(selfplay_best):
            args.checkpoint = selfplay_best
        elif os.path.exists(distill_best):
            args.checkpoint = distill_best
        else:
            print("No checkpoint found. Provide --checkpoint path/to/model.pt")
            sys.exit(1)
    print(f"Loading checkpoint: {args.checkpoint}")

    from mcts_bot import MCTSBot
    ai = MCTSBot(model_path=args.checkpoint, n_sims=args.n_sims)
    print(f"MCTSBot ready ({args.n_sims} sims, device={ai.device})")

    game_cls = ToroidalHexGame if args.torus else HexGame

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    grid_label = "Torus 25x25" if args.torus else "Infinite"
    pygame.display.set_caption(f"KrakenBot \u2014 Hex 6-in-a-Row ({grid_label})")
    clock = pygame.time.Clock()

    fonts = (
        pygame.font.SysFont("Arial", 28, bold=True),
        pygame.font.SysFont("Arial", 20),
        pygame.font.SysFont("Arial", 16),
    )

    # --- Base position ---
    base_board = None
    base_player = Player.A
    if args.position:
        with open(args.position, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "board" in data:
            base_board = dict(data["board"])
            base_player = data["current_player"]

    game = game_cls(win_length=6)

    if base_board:
        game.board = dict(base_board)
        game.current_player = base_player
        game.move_count = len(base_board)
        game.moves_left_in_turn = 2 if base_board else 1

    human_player = Player.A
    hover_hex = None
    last_ai_time = 0
    ai_stats = None
    last_ai_moves = ()
    edit_hover_btn = 1
    show_numbers = False
    move_numbers = {}
    turn_number = 0
    save_msg = None
    save_msg_time = 0
    autoplay = False
    move_history = []
    history_pos = 0

    # Pre-load moves -> REVIEW mode
    if args.moves:
        for token in args.moves.split():
            q, r = token.split(",")
            move_history.append((int(q), int(r)))
        show_numbers = True
        mode = MODE_REVIEW
    else:
        mode = MODE_PLAY

    def _rebuild(pos):
        return rebuild_game(move_history[:pos], base_board, base_player,
                            game_cls=game_cls)

    while True:
        now = pygame.time.get_ticks()

        visible_cells = get_visible_cells_torus() if args.torus else get_visible_cells(game)
        hex_size, ox, oy = compute_view(visible_cells)

        if save_msg and now - save_msg_time > 2000:
            save_msg = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEMOTION:
                q, r = pixel_to_hex(*event.pos, hex_size, ox, oy)
                if args.torus:
                    q, r = q % TORUS_SIZE, r % TORUS_SIZE
                if mode == MODE_EDIT:
                    hover_hex = (q, r) if (q, r) in visible_cells else None
                elif mode == MODE_PLAY and game.current_player == human_player and not game.game_over:
                    hover_hex = (q, r) if ((q, r) in visible_cells and game.is_valid_move(q, r)) else None
                else:
                    hover_hex = None

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if mode == MODE_EDIT:
                    q, r = pixel_to_hex(*event.pos, hex_size, ox, oy)
                    if args.torus:
                        q, r = q % TORUS_SIZE, r % TORUS_SIZE
                    if event.button == 1:
                        if game.board.get((q, r)) == Player.A:
                            del game.board[(q, r)]
                        else:
                            game.board[(q, r)] = Player.A
                    elif event.button == 3:
                        if game.board.get((q, r)) == Player.B:
                            del game.board[(q, r)]
                        else:
                            game.board[(q, r)] = Player.B
                    edit_hover_btn = event.button

                elif mode == MODE_PLAY and event.button == 1:
                    if game.current_player == human_player and not game.game_over:
                        q, r = pixel_to_hex(*event.pos, hex_size, ox, oy)
                        if args.torus:
                            q, r = q % TORUS_SIZE, r % TORUS_SIZE
                        if (q, r) in visible_cells and game.is_valid_move(q, r):
                            new_turn = game.moves_left_in_turn == 2 or game.move_count == 0
                            if game.make_move(q, r):
                                move_history = move_history[:history_pos]
                                move_history.append((q, r))
                                history_pos = len(move_history)
                                if new_turn:
                                    turn_number += 1
                                move_numbers[(q, r)] = turn_number
                                hover_hex = None
                            last_ai_time = now

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_n:
                    show_numbers = not show_numbers
                elif event.key == pygame.K_s:
                    pos_dir = os.path.join(os.path.dirname(__file__), "positions")
                    os.makedirs(pos_dir, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_player = Player.A if mode == MODE_EDIT else game.current_player
                    filename = f"position_{timestamp}.pkl"
                    save_path = os.path.join(pos_dir, filename)
                    counter = 1
                    while os.path.exists(save_path):
                        filename = f"position_{timestamp}_{counter}.pkl"
                        save_path = os.path.join(pos_dir, filename)
                        counter += 1
                    position = {
                        "board": dict(game.board),
                        "current_player": save_player,
                        "move_count": len(game.board),
                    }
                    with open(save_path, "wb") as f:
                        pickle.dump(position, f)
                    save_msg = f"Saved {filename}"
                    save_msg_time = now

                elif event.key == pygame.K_r:
                    game = game_cls(win_length=6)
                    if base_board:
                        game.board = dict(base_board)
                        game.current_player = base_player
                        game.move_count = len(base_board)
                        game.moves_left_in_turn = 2 if base_board else 1
                    human_player = Player.A
                    hover_hex = None
                    ai_stats = None
                    last_ai_moves = ()
                    move_numbers = {}
                    turn_number = 0
                    move_history = []
                    history_pos = 0
                    autoplay = False
                    mode = MODE_PLAY

                elif event.key == pygame.K_LEFT and mode != MODE_EDIT:
                    if history_pos > 0:
                        history_pos -= 1
                        game, move_numbers, turn_number = _rebuild(history_pos)
                        ai_stats = None
                        last_ai_moves = ()
                        hover_hex = None
                    mode = MODE_REVIEW
                    autoplay = False

                elif event.key == pygame.K_RIGHT and mode != MODE_EDIT:
                    if history_pos < len(move_history):
                        history_pos += 1
                        game, move_numbers, turn_number = _rebuild(history_pos)
                        ai_stats = None
                        last_ai_moves = ()
                        hover_hex = None
                    mode = MODE_REVIEW
                    autoplay = False

                elif event.key == pygame.K_e:
                    if mode == MODE_EDIT:
                        game.current_player = Player.A
                        game.moves_left_in_turn = 1 if not game.board else 2
                        game.move_count = len(game.board)
                        game.winner = Player.NONE
                        if hasattr(game, 'winning_cells'):
                            game.winning_cells = []
                        game.game_over = False
                        human_player = Player.B
                        last_ai_time = now
                        ai_stats = None
                        last_ai_moves = ()
                        move_numbers = {}
                        turn_number = 0
                        move_history = []
                        history_pos = 0
                        base_board = dict(game.board)
                        base_player = Player.A
                        mode = MODE_PLAY
                    else:
                        mode = MODE_EDIT
                    hover_hex = None

                elif event.key == pygame.K_p and mode == MODE_REVIEW:
                    human_player = game.current_player
                    last_ai_time = now
                    mode = MODE_PLAY

                elif event.key == pygame.K_a and mode == MODE_PLAY and not game.game_over:
                    autoplay = not autoplay
                    last_ai_time = now

                elif event.key == pygame.K_SPACE and mode == MODE_PLAY and not game.game_over:
                    human_player = Player.B if human_player == Player.A else Player.A
                    last_ai_time = now
                    hover_hex = None

        # --- AI turn (PLAY mode only) ---
        if (mode == MODE_PLAY
                and (autoplay or game.current_player != human_player)
                and not game.game_over
                and now - last_ai_time >= AI_MOVE_DELAY):
            draw_board(screen, game, visible_cells, None, hex_size, ox, oy, fonts,
                       mode=mode, human_player=human_player, ai_stats=ai_stats,
                       last_ai_moves=last_ai_moves,
                       show_numbers=show_numbers, move_numbers=move_numbers,
                       save_msg=save_msg, autoplay=autoplay)
            result = ai.get_move(game)
            last_ai_moves = tuple(tuple(m) for m in result)
            turn_number += 1
            for q, r in result:
                if not game.game_over:
                    game.make_move(q, r)
                    move_numbers[(q, r)] = turn_number
                    move_history = move_history[:history_pos]
                    move_history.append((q, r))
                    history_pos = len(move_history)
            ai_stats = (ai.n_sims,)
            last_ai_time = pygame.time.get_ticks()

        draw_board(screen, game, visible_cells, hover_hex, hex_size, ox, oy, fonts,
                   mode=mode, human_player=human_player, ai_stats=ai_stats,
                   last_ai_moves=last_ai_moves, edit_hover_btn=edit_hover_btn,
                   show_numbers=show_numbers, move_numbers=move_numbers,
                   save_msg=save_msg, autoplay=autoplay,
                   review_pos=history_pos, review_total=len(move_history))
        clock.tick(60)


if __name__ == "__main__":
    main()
