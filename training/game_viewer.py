"""Live web dashboard for monitoring self-play games during training.

Starts an HTTP server on a background daemon thread that serves a single-page
dashboard. The grid shows all game slots color-coded by eval; clicking one
opens a live hex board viewer with top MCTS move candidates and a timeline
scrubber to step through the game history. Finished games are kept for review.

Usage: Created by train_loop.py with --viewer flag.
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from mcts.model import BOARD_SIZE

MAX_FINISHED = 2000


# ---------------------------------------------------------------------------
# JSON snapshot helpers (called from HTTP thread, reads shared state)
# ---------------------------------------------------------------------------

def _slot_summary(slot, idx):
    """Lightweight slot summary (no board data)."""
    try:
        g = slot.game
        t = slot.tree
        return {
            "id": idx,
            "gid": slot.game_id,
            "turn": slot.turn_number,
            "moves": g.move_count,
            "player": g.current_player.value,
            "over": g.game_over,
            "winner": g.winner.value,
            "eval": round(t.root_value, 3) if t else 0.0,
        }
    except Exception:
        return {"id": idx, "gid": -1, "turn": 0, "moves": 0,
                "player": 0, "over": False, "winner": 0, "eval": 0.0}


def _slot_detail(slot, idx):
    """Full slot state with board, top MCTS moves, and game history."""
    s = _slot_summary(slot, idx)
    try:
        g = slot.game
        s["board"] = {f"{q},{r}": p.value for (q, r), p in g.board.items()}
        s["moves_left"] = g.moves_left_in_turn

        tree = slot.tree
        if tree and tree.root_pos and tree.root_pos.move_node.actions:
            root = tree.root_pos.move_node
            top = []
            for i in range(root.n):
                v = root.visits[i]
                if v > 0:
                    a = root.actions[i]
                    top.append({
                        "q": a // BOARD_SIZE, "r": a % BOARD_SIZE,
                        "v": v,
                        "qv": round(root.values[i] / v, 3),
                    })
            top.sort(key=lambda x: x["v"], reverse=True)
            s["top"] = top[:8]
        else:
            s["top"] = []

        s["history"] = [
            {"b": ex["board"], "p": ex["current_player"]}
            for ex in slot.examples
        ]
    except Exception:
        s["board"] = {}
        s["top"] = []
        s["moves_left"] = 0
        s["history"] = []
    return s


def _finished_summary(info, idx):
    """Compact summary for a finished game."""
    return {
        "id": idx,
        "gid": info["gid"],
        "winner": info["winner"],
        "moves": info["moves"],
        "turns": info["turns"],
    }


def _finished_detail(info, idx):
    """Full detail for a finished game (same shape as _slot_detail)."""
    return {
        "id": idx,
        "gid": info["gid"],
        "turn": info["turns"],
        "moves": info["moves"],
        "player": info["winner"],
        "over": True,
        "winner": info["winner"],
        "eval": 0.0,
        "board": info["board"],
        "moves_left": 0,
        "top": [],
        "history": info["history"],
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    _state = None
    _html = None

    def log_message(self, *_):
        pass

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/":
            self._send(200, "text/html", self._html.encode())
        elif p.path == "/api/state":
            qs = parse_qs(p.query)
            sel_slot = int(qs["selected"][0]) if "selected" in qs else None
            sel_fin = int(qs["finished"][0]) if "finished" in qs else None
            self._serve_state(sel_slot, sel_fin)
        else:
            self.send_error(404)

    def _serve_state(self, sel_slot, sel_fin):
        st = self._state
        slots = st.slots
        data = {
            "completed": st.games_completed,
            "total": st.games_total,
            "round": st.round_id,
            "torus": BOARD_SIZE,
            "slots": [_slot_summary(s, i) for i, s in enumerate(slots)],
            "finished": [_finished_summary(f, i)
                         for i, f in enumerate(st.finished)],
        }
        if sel_fin is not None and 0 <= sel_fin < len(st.finished):
            data["sel"] = _finished_detail(st.finished[sel_fin], sel_fin)
        elif sel_slot is not None and 0 <= sel_slot < len(slots):
            data["sel"] = _slot_detail(slots[sel_slot], sel_slot)
        self._send(200, "application/json", json.dumps(data).encode())

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Shared mutable state
# ---------------------------------------------------------------------------

class _ViewerState:
    __slots__ = ("slots", "games_completed", "games_total", "round_id",
                 "finished")

    def __init__(self):
        self.slots = []
        self.games_completed = 0
        self.games_total = 0
        self.round_id = 0
        self.finished = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GameViewer:
    """Live web dashboard for self-play monitoring.

    Call start() once, then update_slots() each turn from the training thread.
    """

    def __init__(self, port=8765):
        self.port = port
        self._state = _ViewerState()
        self._server = None

    def start(self):
        handler = type("H", (_Handler,), {
            "_state": self._state,
            "_html": _VIEWER_HTML,
        })
        self._server = HTTPServer(("0.0.0.0", self.port), handler)
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()
        print(f"\n  Game viewer: http://localhost:{self.port}\n")

    def update_slots(self, slots, games_completed, games_total, round_id):
        """Called from training thread after each turn."""
        self._state.slots = slots[:]
        self._state.games_completed = games_completed
        self._state.games_total = games_total
        self._state.round_id = round_id

    def add_finished(self, slot):
        """Snapshot a finished game for later review. Called from training thread."""
        self._state.finished.append({
            "gid": slot.game_id,
            "winner": slot.game.winner.value,
            "moves": slot.game.move_count,
            "turns": slot.turn_number,
            "board": {f"{q},{r}": p.value
                      for (q, r), p in slot.game.board.items()},
            "history": [{"b": ex["board"], "p": ex["current_player"]}
                        for ex in slot.examples],
        })
        fin = self._state.finished
        if len(fin) > MAX_FINISHED:
            del fin[:len(fin) - MAX_FINISHED]

    def add_finished_data(self, gid, winner, moves, turns, board, history):
        """Add a finished game from pre-built data (parallel self-play)."""
        self._state.finished.append({
            "gid": gid,
            "winner": winner,
            "moves": moves,
            "turns": turns,
            "board": board,
            "history": history,
        })
        fin = self._state.finished
        if len(fin) > MAX_FINISHED:
            del fin[:len(fin) - MAX_FINISHED]

    def stop(self):
        if self._server:
            self._server.shutdown()


# ---------------------------------------------------------------------------
# Single-page dashboard HTML
# ---------------------------------------------------------------------------

_VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HexTTT Live</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{
  background:#0d1117;color:#c9d1d9;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,sans-serif;
  display:flex;flex-direction:column;
}
body::before{
  content:'';display:block;height:3px;flex-shrink:0;
  background:linear-gradient(90deg,#58a6ff,#3fb950,#f0883e,#f85149);
}

header{
  display:flex;align-items:center;gap:24px;
  padding:10px 20px;background:#161b22;
  border-bottom:1px solid #30363d;flex-shrink:0;
}
.logo{
  font-size:18px;font-weight:700;
  background:linear-gradient(135deg,#58a6ff,#f0883e);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}
.stats{display:flex;align-items:center;gap:18px;font-size:13px;color:#8b949e}
.stats b{color:#c9d1d9;font-weight:600}
.pbar{width:180px;height:5px;background:#21262d;border-radius:3px;overflow:hidden}
.pfill{height:100%;border-radius:3px;transition:width .3s;
  background:linear-gradient(90deg,#58a6ff,#3fb950)}

main{display:flex;flex:1;overflow:hidden}

/* grid panel */
#gp{
  width:auto;max-width:580px;border-right:1px solid #30363d;
  padding:12px;overflow-y:auto;flex-shrink:0;
}
#gp h3{
  font-size:11px;color:#8b949e;text-transform:uppercase;
  letter-spacing:.8px;margin-bottom:8px;font-weight:500;
}
#grid{display:grid;gap:2px}
.c{
  width:30px;height:30px;border-radius:4px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  font-size:10px;color:rgba(255,255,255,.5);font-weight:600;
  transition:transform .1s,box-shadow .15s;position:relative;
}
.c:hover{transform:scale(1.2);z-index:1;box-shadow:0 0 10px rgba(88,166,255,.35)}
.c.sel{outline:2px solid #58a6ff;outline-offset:1px;box-shadow:0 0 14px rgba(88,166,255,.45)}
.c.done{opacity:.4}
.dt{position:absolute;top:2px;right:2px;width:5px;height:5px;border-radius:50%}

/* finished list */
#fp{margin-top:14px;border-top:1px solid #30363d;padding-top:8px}
#flist{max-height:260px;overflow-y:auto;margin-top:4px}
.fr{
  display:flex;gap:8px;align-items:center;padding:3px 8px;
  cursor:pointer;border-radius:3px;font-size:12px;
  font-variant-numeric:tabular-nums;
}
.fr:hover{background:#21262d}
.fr.sel{background:#1f3a5f;outline:1px solid rgba(88,166,255,.3)}

/* board panel */
#bp{flex:1;display:flex;flex-direction:column;padding:16px 24px;gap:10px;overflow:hidden;min-width:0}
#bp h2{font-size:20px;font-weight:600}
#info{display:flex;gap:18px;font-size:13px;color:#8b949e;flex-wrap:wrap;align-items:center}
#info .v{color:#c9d1d9;font-weight:600;font-variant-numeric:tabular-nums}
.tag{padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;display:inline-block}
.tA{background:rgba(88,166,255,.15);color:#58a6ff}
.tB{background:rgba(240,136,62,.15);color:#f0883e}
.ebar{
  width:140px;height:6px;border-radius:3px;position:relative;
  background:linear-gradient(90deg,#f85149 0%,#30363d 50%,#3fb950 100%);
}
.emk{
  position:absolute;top:-4px;width:4px;height:14px;
  background:#fff;border-radius:2px;transform:translateX(-50%);
  transition:left .3s;
}
#bw{flex:1;display:flex;align-items:center;justify-content:center;min-height:0}
#bsvg{width:100%;height:100%;max-width:900px;max-height:750px}
#ph{color:#8b949e;font-size:15px}
#mv{display:flex;gap:6px;flex-wrap:wrap}
.mc{
  padding:3px 8px;background:#21262d;border-radius:5px;
  font-size:11px;display:flex;gap:5px;font-variant-numeric:tabular-nums;
}
.mc .cd{color:#58a6ff;font-weight:600}
.mc .vs{color:#8b949e}

/* timeline */
#tl{
  display:none;align-items:center;gap:6px;padding:4px 0;flex-shrink:0;
}
#tl button{
  background:#21262d;border:1px solid #30363d;color:#c9d1d9;
  border-radius:4px;padding:2px 8px;cursor:pointer;font-size:14px;
  line-height:1.3;
}
#tl button:hover{background:#30363d}
#tl button.on{background:#1f3a5f;border-color:#58a6ff;color:#58a6ff}
#tsl{flex:1;accent-color:#58a6ff;max-width:320px}
#tlbl{font-size:12px;color:#8b949e;min-width:100px}
</style>
</head>
<body>

<header>
  <div class="logo">HexTTT Live</div>
  <div class="stats">
    <span>Round <b id="rnd">--</b></span>
    <span>Games <b id="gcnt">--</b></span>
    <div class="pbar"><div class="pfill" id="prog"></div></div>
    <span id="gps" style="min-width:80px"></span>
  </div>
</header>

<main>
  <div id="gp">
    <h3 id="gtitle">Game Slots</h3>
    <div id="grid"></div>
    <div id="fp">
      <h3 id="ftitle">Finished (0)</h3>
      <div id="flist"></div>
    </div>
  </div>
  <div id="bp">
    <h2 id="title">Select a game</h2>
    <div id="info"></div>
    <div id="bw">
      <div id="ph">Click any slot to view its board</div>
      <svg id="bsvg" style="display:none" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
    <div id="tl">
      <button id="tb0" title="First turn (Home)">&#x23EE;</button>
      <button id="tbp" title="Previous turn (Left)">&#x25C0;</button>
      <input type="range" id="tsl" min="0" max="0" value="0">
      <button id="tbn" title="Next turn (Right)">&#x25B6;</button>
      <button id="tbe" title="Latest turn (End)">&#x23ED;</button>
      <button id="tbl" class="on" title="Follow live">LIVE</button>
      <span id="tlbl"></span>
    </div>
    <div id="mv"></div>
  </div>
</main>

<script>
/* ---- state ---- */
let selSlot = null;   // selected active slot index
let selFin = null;    // selected finished game index
let gridN = 0;
let TORUS = 25;
let prevCompleted = 0, prevT = Date.now(), gps = 0;
let viewMode = 'live';
let histData = [];
let curGameId = -1;
let prevFinLen = -1;

/* ---- helpers ---- */
function evalBg(ev, player, over, w) {
  if (over) {
    if (w===1) return 'rgba(88,166,255,.25)';
    if (w===2) return 'rgba(240,136,62,.25)';
    return 'rgba(128,128,128,.18)';
  }
  // convert to absolute: positive = A winning, negative = B winning
  const evA = player === 1 ? ev : -ev;
  const a = .08 + Math.min(Math.abs(evA),1) * .4;
  if (evA > .03) return 'rgba(88,166,255,' + a + ')';    // A winning
  if (evA <-.03) return 'rgba(240,136,62,' + a + ')';    // B winning
  return 'rgba(128,128,128,.12)';
}

/* ---- active grid ---- */
function initGrid(n) {
  const g = document.getElementById('grid');
  g.innerHTML = '';
  const cols = Math.min(Math.ceil(Math.sqrt(n)), 20);
  g.style.gridTemplateColumns = 'repeat(' + cols + ', 30px)';
  for (let i = 0; i < n; i++) {
    const d = document.createElement('div');
    d.className = 'c';
    d.dataset.i = i;
    d.onclick = () => {
      if (selSlot !== i) { viewMode = 'live'; curGameId = -1; }
      selSlot = i; selFin = null;
      document.querySelectorAll('.c.sel').forEach(x => x.classList.remove('sel'));
      d.classList.add('sel');
    };
    g.appendChild(d);
  }
  gridN = n;
  document.getElementById('gtitle').textContent = 'Game Slots (' + n + ')';
}

function updateGrid(slots) {
  if (slots.length !== gridN) initGrid(slots.length);
  const cells = document.querySelectorAll('.c');
  for (let i = 0; i < slots.length && i < cells.length; i++) {
    const s = slots[i], el = cells[i];
    el.style.background = evalBg(s.eval, s.player, s.over, s.winner);
    el.className = 'c' + (i === selSlot ? ' sel' : '') + (s.over ? ' done' : '');
    el.title = 'Slot ' + i + ' | Game #' + s.gid + '\nTurn ' + s.turn + ' | ' + s.moves + ' moves\nEval: ' + (s.eval>0?'+':'') + s.eval.toFixed(3) + (s.over ? '\nGame Over' : '');
    el.innerHTML = '<div class="dt" style="background:' + (s.player===1?'#58a6ff':'#f0883e') + '"></div><span>' + s.turn + '</span>';
  }
}

/* ---- finished list ---- */
function updateFinished(finished) {
  document.getElementById('ftitle').textContent = 'Finished (' + finished.length + ')';
  const el = document.getElementById('flist');
  if (finished.length !== prevFinLen) {
    prevFinLen = finished.length;
    const rev = finished.slice().reverse();
    el.innerHTML = rev.map(f => {
      const wc = f.winner===1 ? '#58a6ff' : f.winner===2 ? '#f0883e' : '#8b949e';
      const wt = f.winner===1 ? 'A' : f.winner===2 ? 'B' : '=';
      return '<div class="fr" data-idx="' + f.id + '">' +
        '<span style="color:' + wc + ';font-weight:600;min-width:42px">#' + f.gid + '</span>' +
        '<span style="color:' + wc + ';min-width:18px">' + wt + '</span>' +
        '<span style="color:#8b949e">' + f.moves + 'mv ' + f.turns + 't</span>' +
      '</div>';
    }).join('');
  }
  el.querySelectorAll('.fr').forEach(fr => {
    fr.classList.toggle('sel', selFin === parseInt(fr.dataset.idx));
  });
}

document.getElementById('flist').addEventListener('click', e => {
  const fr = e.target.closest('.fr');
  if (!fr) return;
  selFin = parseInt(fr.dataset.idx);
  selSlot = null;
  viewMode = 'live'; curGameId = -1;
  document.querySelectorAll('.c.sel').forEach(x => x.classList.remove('sel'));
});

/* ---- header ---- */
function updateHeader(d) {
  document.getElementById('rnd').textContent = d.round;
  document.getElementById('gcnt').textContent = d.completed + ' / ' + d.total;
  const pct = d.total > 0 ? (100 * d.completed / d.total) : 0;
  document.getElementById('prog').style.width = pct + '%';
  const now = Date.now(), dt = (now - prevT) / 1000;
  if (dt > 2) {
    gps = (d.completed - prevCompleted) / dt;
    prevCompleted = d.completed;
    prevT = now;
  }
  document.getElementById('gps').textContent = gps > 0 ? gps.toFixed(1) + ' games/s' : '';
}

/* ---- hex math ---- */
const S3 = Math.sqrt(3);
const HEX6 = [[1,0],[0,1],[-1,1],[-1,0],[0,-1],[1,-1]];

function h2p(q, r, sz) { return [sz*S3*(q+.5*r), sz*1.5*r]; }

function hpts(cx, cy, sz) {
  let p = '';
  for (let i = 0; i < 6; i++) {
    const a = Math.PI/6 + Math.PI/3*i;
    if (i) p += ' ';
    p += (cx+sz*Math.cos(a)).toFixed(1) + ',' + (cy+sz*Math.sin(a)).toFixed(1);
  }
  return p;
}

/* ---- full-grid hex drawing ---- */
function drawHex(boardDict, topMoves, player) {
  const svg = document.getElementById('bsvg');
  const bmap = {};
  for (const [k, pl] of Object.entries(boardDict)) bmap[k] = pl;

  const topSet = new Map();
  if (topMoves) for (const m of topMoves) topSet.set(m.q + ',' + m.r, m);

  // candidate color = current player
  const isA = player === 1;
  const candFill   = isA ? 'rgba(88,166,255,.13)' : 'rgba(240,136,62,.13)';
  const candStroke = isA ? 'rgba(88,166,255,.45)' : 'rgba(240,136,62,.45)';
  const dotFill    = isA ? 'rgba(88,166,255,.55)' : 'rgba(240,136,62,.55)';
  const dotStroke  = isA ? '#58a6ff' : '#f0883e';

  const sz = 12, szH = sz * .88;

  // viewBox from grid corners
  let x0=1e9, y0=1e9, x1=-1e9, y1=-1e9;
  for (const [q,r] of [[0,0],[TORUS-1,0],[0,TORUS-1],[TORUS-1,TORUS-1]]) {
    const [px,py] = h2p(q,r,sz);
    if (px<x0) x0=px; if (py<y0) y0=py;
    if (px>x1) x1=px; if (py>y1) y1=py;
  }
  const pad = sz*2;
  svg.setAttribute('viewBox', (x0-pad)+' '+(y0-pad)+' '+(x1-x0+pad*2)+' '+(y1-y0+pad*2));

  let html = '';

  // render full torus grid
  for (let q = 0; q < TORUS; q++) {
    for (let r = 0; r < TORUS; r++) {
      const [px, py] = h2p(q, r, sz);
      const k = q + ',' + r;
      const pl = bmap[k] || 0;
      const tm = topSet.get(k);
      let fill, stroke, sw;
      if (pl === 1) {
        fill = '#58a6ff'; stroke = '#3a6fbf'; sw = '1.2';
      } else if (pl === 2) {
        fill = '#f0883e'; stroke = '#c55522'; sw = '1.2';
      } else if (tm) {
        fill = candFill; stroke = candStroke; sw = '1';
      } else {
        fill = '#0f1318'; stroke = '#1a2030'; sw = '.3';
      }
      html += '<polygon points="' + hpts(px,py,szH) + '" fill="' + fill +
        '" stroke="' + stroke + '" stroke-width="' + sw +
        '"><title>(' + q + ',' + r + ')' +
        (pl ? ' Player ' + (pl===1?'A':'B') : '') +
        '</title></polygon>';
    }
  }

  // top-move dots
  if (topMoves && topMoves.length) {
    const maxV = topMoves[0].v;
    for (const m of topMoves) {
      const [px, py] = h2p(m.q, m.r, sz);
      const rad = 1.5 + 3 * (m.v / maxV);
      html += '<circle cx="' + px.toFixed(1) + '" cy="' + py.toFixed(1) +
        '" r="' + rad.toFixed(1) + '" fill="' + dotFill +
        '" stroke="' + dotStroke + '" stroke-width=".5"><title>(' +
        m.q + ',' + m.r + ') visits=' + m.v + ' Q=' +
        (m.qv>0?'+':'') + m.qv + '</title></circle>';
    }
  }

  svg.innerHTML = html;
}

/* ---- timeline helpers ---- */
function goLive() {
  viewMode = 'live';
  document.getElementById('tbl').classList.add('on');
}
function goToTurn(t) {
  viewMode = Math.max(0, Math.min(t, Math.max(histData.length - 1, 0)));
  document.getElementById('tbl').classList.remove('on');
}
function stepTurn(d) {
  const max = histData.length;
  const cur = viewMode === 'live' ? max : viewMode;
  const next = cur + d;
  if (next >= max) goLive();
  else if (next < 0) goToTurn(0);
  else goToTurn(next);
}

/* ---- board + info rendering ---- */
function renderBoard(sel) {
  const svg = document.getElementById('bsvg');
  const ph = document.getElementById('ph');
  const tl = document.getElementById('tl');

  if (!sel || !sel.board) {
    svg.style.display='none'; ph.style.display=''; tl.style.display='none';
    return;
  }
  svg.style.display=''; ph.style.display='none'; tl.style.display='flex';

  if (sel.gid !== curGameId) { curGameId = sel.gid; goLive(); }

  histData = sel.history || [];
  const maxT = histData.length;
  const slider = document.getElementById('tsl');
  slider.max = maxT;

  let bd, tm, isLive, dispPlayer;
  if (viewMode === 'live') {
    bd = sel.board; tm = sel.top; isLive = true; dispPlayer = sel.player;
    slider.value = maxT;
  } else {
    const idx = Math.min(viewMode, maxT - 1);
    if (idx >= 0 && idx < maxT) {
      const h = histData[idx];
      bd = JSON.parse(h.b); dispPlayer = h.p;
    } else {
      bd = sel.board; dispPlayer = sel.player;
    }
    tm = null; isLive = false;
    slider.value = typeof viewMode === 'number' ? viewMode : maxT;
  }

  drawHex(bd, tm, dispPlayer);

  // timeline label
  const tlbl = document.getElementById('tlbl');
  if (isLive) {
    tlbl.innerHTML = '<b style="color:#3fb950">LIVE</b> turn ' + sel.turn;
  } else {
    tlbl.textContent = 'Turn ' + viewMode + ' / ' + maxT;
  }

  // info bar
  const info = document.getElementById('info');
  const pTag = dispPlayer===1
    ? '<span class="tag tA">A</span>'
    : '<span class="tag tB">B</span>';
  document.getElementById('title').textContent = 'Game #' + sel.gid;

  if (isLive) {
    // convert eval to A's perspective: positive = A winning, negative = B winning
    const evA = sel.player === 1 ? sel.eval : -sel.eval;
    const evSign = evA >= 0 ? '+' : '';
    const evCol = evA > .1 ? '#58a6ff' : evA < -.1 ? '#f0883e' : '#c9d1d9';
    const evPct = ((evA + 1) / 2 * 100).toFixed(0);
    let status = sel.over
      ? (sel.winner===1 ? 'Winner: A' : sel.winner===2 ? 'Winner: B' : 'Draw')
      : 'Playing';
    info.innerHTML =
      '<span>Turn <span class="v">' + sel.turn + '</span></span>' +
      '<span>Moves <span class="v">' + sel.moves + '</span></span>' +
      '<span>Player ' + pTag + '</span>' +
      '<span>Eval <span class="v" style="color:' + evCol + '">' + evSign + evA.toFixed(3) + '</span></span>' +
      '<div class="ebar"><div class="emk" style="left:' + evPct + '%"></div></div>' +
      '<span class="v">' + status + '</span>';
  } else {
    info.innerHTML =
      '<span>Turn <span class="v">' + viewMode + ' / ' + maxT + '</span></span>' +
      '<span>Player ' + pTag + '</span>' +
      '<span class="v" style="color:#8b949e">Viewing history</span>';
  }

  const mvEl = document.getElementById('mv');
  if (isLive && sel.top && sel.top.length) {
    mvEl.innerHTML = sel.top.map(m => {
      const qc = m.qv > .05 ? '#3fb950' : m.qv < -.05 ? '#f85149' : '#c9d1d9';
      return '<div class="mc"><span class="cd">(' + m.q + ',' + m.r + ')</span><span class="vs">' + m.v + 'v</span><span style="color:' + qc + '">' + (m.qv>0?'+':'') + m.qv.toFixed(2) + '</span></div>';
    }).join('');
  } else {
    mvEl.innerHTML = '';
  }
}

/* ---- timeline controls ---- */
document.getElementById('tb0').onclick = () => goToTurn(0);
document.getElementById('tbp').onclick = () => stepTurn(-1);
document.getElementById('tbn').onclick = () => stepTurn(1);
document.getElementById('tbe').onclick = () => goLive();
document.getElementById('tbl').onclick = () => goLive();
document.getElementById('tsl').oninput = e => {
  const v = parseInt(e.target.value);
  if (v >= histData.length) goLive(); else goToTurn(v);
};

document.addEventListener('keydown', e => {
  if ((selSlot === null && selFin === null) || e.target.tagName === 'INPUT') return;
  switch (e.key) {
    case 'ArrowLeft':  e.preventDefault(); stepTurn(-1); break;
    case 'ArrowRight': e.preventDefault(); stepTurn(1);  break;
    case 'Home':       e.preventDefault(); goToTurn(0);  break;
    case 'End':        e.preventDefault(); goLive();     break;
  }
});

/* ---- main loop ---- */
async function tick() {
  try {
    let url = '/api/state';
    if (selFin !== null) url += '?finished=' + selFin;
    else if (selSlot !== null) url += '?selected=' + selSlot;
    const r = await fetch(url);
    if (!r.ok) return;
    const d = await r.json();
    TORUS = d.torus || 25;
    updateHeader(d);
    updateGrid(d.slots);
    updateFinished(d.finished || []);
    renderBoard(d.sel || null);
  } catch(e) {}
}
setInterval(tick, 400);
tick();
</script>
</body>
</html>
"""
