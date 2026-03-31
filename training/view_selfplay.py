"""Offline browser for saved self-play parquet data.

Loads round_*.parquet from training/data/selfplay/ and serves an
interactive HTML viewer to browse games, step through turns, and see
MCTS visit distributions.

Usage:
    python -m training.view_selfplay [--port 8766] [--data-dir training/data/selfplay]
"""

import argparse
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd

from mcts.model import BOARD_SIZE

N = BOARD_SIZE  # flat index = q * N + r


def _load_rounds(data_dir: Path) -> dict[int, pd.DataFrame]:
    """Load all round_*.parquet files, keyed by round number."""
    rounds = {}
    for f in sorted(data_dir.glob("round_*.parquet")):
        rid = int(f.stem.split("_")[1])
        rounds[rid] = pd.read_parquet(f)
    return rounds


def _game_summary(gdf: pd.DataFrame) -> dict:
    """Summary for a game from its DataFrame rows."""
    first = gdf.iloc[0]
    last = gdf.iloc[-1]
    board = json.loads(last["board"])
    return {
        "game_id": int(first["game_id"]),
        "turns": len(gdf),
        "moves": int(last["move_count"]),
        "value": float(first["value_target"]),
        "drawn": bool(first["game_drawn"]),
    }


def _game_detail(gdf: pd.DataFrame) -> dict:
    """Full game detail with turn-by-turn history."""
    first = gdf.iloc[0]
    turns = []
    for _, row in gdf.iterrows():
        board = json.loads(row["board"])
        pv_raw = json.loads(row["pair_visits"])
        # decode pair visits: key "a,b" -> flat indices -> (q1,r1,q2,r2)
        top_pairs = []
        for k, v in sorted(pv_raw.items(), key=lambda x: -x[1])[:10]:
            a, b = k.split(",")
            a, b = int(a), int(b)
            top_pairs.append({
                "q1": a // N, "r1": a % N,
                "q2": b // N, "r2": b % N,
                "v": v,
            })
        turns.append({
            "board": board,
            "player": int(row["current_player"]),
            "move_count": int(row["move_count"]),
            "moves_left": int(row["moves_left"]),
            "top_pairs": top_pairs,
        })
    return {
        "game_id": int(first["game_id"]),
        "round_id": int(first["round_id"]),
        "value": float(first["value_target"]),
        "drawn": bool(first["game_drawn"]),
        "turns": turns,
    }


class _Handler(BaseHTTPRequestHandler):
    _rounds: dict[int, pd.DataFrame] = None
    _html: str = None
    _game_cache: dict = None  # (round_id, game_id) -> detail dict

    def log_message(self, *_):
        pass

    def do_GET(self):
        p = urlparse(self.path)
        qs = parse_qs(p.query)
        if p.path == "/":
            self._send(200, "text/html", self._html.encode())
        elif p.path == "/api/rounds":
            self._serve_rounds()
        elif p.path == "/api/games":
            rid = int(qs["round"][0])
            self._serve_games(rid)
        elif p.path == "/api/game":
            rid = int(qs["round"][0])
            gid = int(qs["game_id"][0])
            self._serve_game(rid, gid)
        else:
            self.send_error(404)

    def _serve_rounds(self):
        info = []
        for rid in sorted(self._rounds):
            df = self._rounds[rid]
            info.append({
                "round_id": rid,
                "games": int(df["game_id"].nunique()),
                "positions": len(df),
            })
        self._send(200, "application/json", json.dumps(info).encode())

    def _serve_games(self, rid):
        df = self._rounds.get(rid)
        if df is None:
            self._send(200, "application/json", b"[]")
            return
        games = []
        for gid, gdf in df.groupby("game_id"):
            gdf = gdf.sort_values("move_count")
            games.append(_game_summary(gdf))
        games.sort(key=lambda g: g["game_id"])
        self._send(200, "application/json", json.dumps(games).encode())

    def _serve_game(self, rid, gid):
        key = (rid, gid)
        if key in self._game_cache:
            data = self._game_cache[key]
        else:
            df = self._rounds.get(rid)
            if df is None:
                self._send(200, "application/json", b"{}")
                return
            gdf = df[df["game_id"] == gid].sort_values("move_count")
            if gdf.empty:
                self._send(200, "application/json", b"{}")
                return
            data = _game_detail(gdf)
            self._game_cache[key] = data
        self._send(200, "application/json", json.dumps(data).encode())

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Browse self-play data")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-dir", type=str,
                        default="training/data/selfplay")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Loading rounds from {data_dir} ...")
    rounds = _load_rounds(data_dir)
    if not rounds:
        print("No round_*.parquet files found.")
        return
    for rid, df in sorted(rounds.items()):
        print(f"  round {rid}: {df['game_id'].nunique()} games, "
              f"{len(df)} positions")

    handler = type("H", (_Handler,), {
        "_rounds": rounds,
        "_html": _VIEWER_HTML,
        "_game_cache": {},
    })
    server = HTTPServer(("0.0.0.0", args.port), handler)
    print(f"\n  Self-play viewer: http://localhost:{args.port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


# ---------------------------------------------------------------------------
# Single-page viewer HTML
# ---------------------------------------------------------------------------

_VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Self-Play Browser</title>
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

/* round tabs */
.rtabs{display:flex;gap:2px;flex-wrap:wrap}
.rtab{
  padding:3px 12px;border-radius:4px;cursor:pointer;
  font-size:12px;font-weight:600;background:#21262d;color:#8b949e;
  border:1px solid transparent;transition:all .15s;
}
.rtab:hover{background:#30363d;color:#c9d1d9}
.rtab.sel{background:#1f3a5f;color:#58a6ff;border-color:rgba(88,166,255,.3)}

main{display:flex;flex:1;overflow:hidden}

/* game list panel */
#gp{
  width:280px;border-right:1px solid #30363d;
  display:flex;flex-direction:column;flex-shrink:0;
}
#gp h3{
  font-size:11px;color:#8b949e;text-transform:uppercase;
  letter-spacing:.8px;padding:10px 12px 6px;font-weight:500;
}
#filter{
  margin:0 10px 6px;padding:4px 8px;
  background:#0d1117;border:1px solid #30363d;border-radius:4px;
  color:#c9d1d9;font-size:12px;outline:none;
}
#filter:focus{border-color:#58a6ff}
#glist{flex:1;overflow-y:auto;padding:0 4px}
.gr{
  display:flex;gap:6px;align-items:center;padding:4px 8px;
  cursor:pointer;border-radius:4px;font-size:12px;
  font-variant-numeric:tabular-nums;transition:background .1s;
}
.gr:hover{background:#21262d}
.gr.sel{background:#1f3a5f;outline:1px solid rgba(88,166,255,.3)}
.gr .gid{min-width:48px;font-weight:600}
.gr .out{min-width:22px;font-weight:600}
.gr .meta{color:#8b949e}

/* board panel */
#bp{flex:1;display:flex;flex-direction:column;padding:16px 24px;gap:10px;overflow:hidden;min-width:0}
#bp h2{font-size:20px;font-weight:600}
#info{display:flex;gap:18px;font-size:13px;color:#8b949e;flex-wrap:wrap;align-items:center}
#info .v{color:#c9d1d9;font-weight:600;font-variant-numeric:tabular-nums}
.tag{padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;display:inline-block}
.tA{background:rgba(88,166,255,.15);color:#58a6ff}
.tB{background:rgba(240,136,62,.15);color:#f0883e}
.tD{background:rgba(128,128,128,.15);color:#8b949e}
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
#tsl{flex:1;accent-color:#58a6ff;max-width:320px}
#tlbl{font-size:12px;color:#8b949e;min-width:120px}
</style>
</head>
<body>

<header>
  <div class="logo">Self-Play Browser</div>
  <div class="stats">
    <span>Round <b id="rnd">--</b></span>
    <span>Games <b id="gcnt">--</b></span>
    <span>Positions <b id="pcnt">--</b></span>
  </div>
  <div class="rtabs" id="rtabs"></div>
</header>

<main>
  <div id="gp">
    <h3 id="gtitle">Games</h3>
    <input id="filter" type="text" placeholder="Filter by game ID...">
    <div id="glist"></div>
  </div>
  <div id="bp">
    <h2 id="title">Select a game</h2>
    <div id="info"></div>
    <div id="bw">
      <div id="ph">Click any game to view its board</div>
      <svg id="bsvg" style="display:none" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
    <div id="tl">
      <button id="tb0" title="First turn (Home)">&#x23EE;</button>
      <button id="tbp" title="Previous turn (Left)">&#x25C0;</button>
      <input type="range" id="tsl" min="0" max="0" value="0">
      <button id="tbn" title="Next turn (Right)">&#x25B6;</button>
      <button id="tbe" title="Last turn (End)">&#x23ED;</button>
      <span id="tlbl"></span>
    </div>
    <div id="mv"></div>
  </div>
</main>

<script>
const TORUS = """ + str(BOARD_SIZE) + r""";
const N = TORUS;
let rounds = [];
let curRound = null;
let games = [];
let selGameId = null;
let gameDetail = null;
let curTurn = 0;

/* ---- fetch helpers ---- */
async function loadRounds() {
  const r = await fetch('/api/rounds');
  rounds = await r.json();
  const tabs = document.getElementById('rtabs');
  tabs.innerHTML = rounds.map(r =>
    '<div class="rtab" data-rid="' + r.round_id + '">R' + r.round_id +
    ' <span style="color:#8b949e;font-weight:400">(' + r.games + ')</span></div>'
  ).join('');
  tabs.querySelectorAll('.rtab').forEach(t => {
    t.onclick = () => selectRound(parseInt(t.dataset.rid));
  });
  if (rounds.length) selectRound(rounds[0].round_id);
}

async function selectRound(rid) {
  curRound = rid;
  selGameId = null;
  gameDetail = null;
  document.querySelectorAll('.rtab').forEach(t =>
    t.classList.toggle('sel', parseInt(t.dataset.rid) === rid));
  const info = rounds.find(r => r.round_id === rid);
  document.getElementById('rnd').textContent = rid;
  document.getElementById('gcnt').textContent = info ? info.games : '--';
  document.getElementById('pcnt').textContent = info ? info.positions : '--';

  const r = await fetch('/api/games?round=' + rid);
  games = await r.json();
  renderGameList();
  renderBoard();
}

async function selectGame(gid) {
  selGameId = gid;
  document.querySelectorAll('.gr').forEach(g =>
    g.classList.toggle('sel', parseInt(g.dataset.gid) === gid));
  const r = await fetch('/api/game?round=' + curRound + '&game_id=' + gid);
  gameDetail = await r.json();
  curTurn = 0;
  renderBoard();
}

/* ---- game list ---- */
function renderGameList() {
  const el = document.getElementById('glist');
  const filter = document.getElementById('filter').value.trim();
  const filtered = filter
    ? games.filter(g => String(g.game_id).includes(filter))
    : games;
  document.getElementById('gtitle').textContent = 'Games (' + filtered.length + ')';
  el.innerHTML = filtered.map(g => {
    const wc = g.drawn ? '#8b949e' : g.value > 0 ? '#58a6ff' : '#f0883e';
    const wt = g.drawn ? '=' : g.value > 0 ? 'P1' : 'P2';
    return '<div class="gr" data-gid="' + g.game_id + '">' +
      '<span class="gid" style="color:' + wc + '">#' + g.game_id + '</span>' +
      '<span class="out" style="color:' + wc + '">' + wt + '</span>' +
      '<span class="meta">' + g.moves + 'mv ' + g.turns + 't</span>' +
    '</div>';
  }).join('');
  el.querySelectorAll('.gr').forEach(g => {
    g.onclick = () => selectGame(parseInt(g.dataset.gid));
  });
}
document.getElementById('filter').addEventListener('input', renderGameList);

/* ---- hex math ---- */
const S3 = Math.sqrt(3);
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

/* ---- hex drawing ---- */
function drawHex(boardDict, topPairs, player) {
  const svg = document.getElementById('bsvg');
  const sz = 12, szH = sz * .88;

  // Find bounding box from occupied cells (with some padding around them)
  let minQ=Infinity, maxQ=-Infinity, minR=Infinity, maxR=-Infinity;
  for (const k of Object.keys(boardDict)) {
    const [q,r] = k.split(',').map(Number);
    if (q<minQ) minQ=q; if (q>maxQ) maxQ=q;
    if (r<minR) minR=r; if (r>maxR) maxR=r;
  }
  // Also include pair targets in bounding box
  if (topPairs) for (const p of topPairs) {
    for (const q of [p.q1,p.q2]) {
      if (q<minQ) minQ=q; if (q>maxQ) maxQ=q;
    }
    for (const r of [p.r1,p.r2]) {
      if (r<minR) minR=r; if (r>maxR) maxR=r;
    }
  }
  // Pad by 2 cells around occupied area, clamp to board
  const pad = 3;
  const rMinQ = Math.max(0, minQ-pad), rMaxQ = Math.min(N-1, maxQ+pad);
  const rMinR = Math.max(0, minR-pad), rMaxR = Math.min(N-1, maxR+pad);

  // viewBox
  const [x0,y0] = h2p(rMinQ, rMinR, sz);
  const [x1,y1] = h2p(rMaxQ, rMaxR, sz);
  const vpad = sz*2;
  svg.setAttribute('viewBox',
    (x0-vpad)+' '+(y0-vpad)+' '+(x1-x0+vpad*2)+' '+(y1-y0+vpad*2));

  // Collect pair target cells for highlighting
  const pairCells = new Map(); // "q,r" -> total visits
  if (topPairs) for (const p of topPairs) {
    for (const [q,r] of [[p.q1,p.r1],[p.q2,p.r2]]) {
      const k = q+','+r;
      pairCells.set(k, (pairCells.get(k)||0) + p.v);
    }
  }

  const isA = player === 1;
  const candFill   = isA ? 'rgba(88,166,255,.13)' : 'rgba(240,136,62,.13)';
  const candStroke = isA ? 'rgba(88,166,255,.45)' : 'rgba(240,136,62,.45)';

  let html = '';

  for (let q = rMinQ; q <= rMaxQ; q++) {
    for (let r = rMinR; r <= rMaxR; r++) {
      const [px, py] = h2p(q, r, sz);
      const k = q + ',' + r;
      const pl = boardDict[k] || 0;
      const isPairTarget = pairCells.has(k);
      let fill, stroke, sw;
      if (pl === 1) {
        fill = '#58a6ff'; stroke = '#3a6fbf'; sw = '1.2';
      } else if (pl === 2) {
        fill = '#f0883e'; stroke = '#c55522'; sw = '1.2';
      } else if (isPairTarget) {
        fill = candFill; stroke = candStroke; sw = '1';
      } else {
        fill = '#0f1318'; stroke = '#1a2030'; sw = '.3';
      }
      html += '<polygon points="' + hpts(px,py,szH) + '" fill="' + fill +
        '" stroke="' + stroke + '" stroke-width="' + sw +
        '"><title>(' + q + ',' + r + ')' +
        (pl ? ' Player ' + (pl===1?'A':'B') : '') +
        (isPairTarget ? ' visits=' + pairCells.get(k) : '') +
        '</title></polygon>';
    }
  }

  // Draw pair connections as lines
  if (topPairs && topPairs.length) {
    const maxV = topPairs[0].v;
    for (const p of topPairs) {
      const [x1,y1] = h2p(p.q1, p.r1, sz);
      const [x2,y2] = h2p(p.q2, p.r2, sz);
      const alpha = .15 + .6 * (p.v / maxV);
      const w = .5 + 2.5 * (p.v / maxV);
      const col = isA ? '88,166,255' : '240,136,62';
      html += '<line x1="'+x1.toFixed(1)+'" y1="'+y1.toFixed(1)+
        '" x2="'+x2.toFixed(1)+'" y2="'+y2.toFixed(1)+
        '" stroke="rgba('+col+','+alpha.toFixed(2)+')" stroke-width="'+w.toFixed(1)+
        '" stroke-linecap="round"><title>(' +
        p.q1+','+p.r1+')-('+p.q2+','+p.r2+') v='+p.v+'</title></line>';
    }
    // Dots on pair cells
    for (const p of topPairs) {
      for (const [q,r] of [[p.q1,p.r1],[p.q2,p.r2]]) {
        if (boardDict[q+','+r]) continue; // don't draw over placed stones
        const [px,py] = h2p(q,r,sz);
        const rad = 1.5 + 2.5 * (p.v / maxV);
        const dotFill = isA ? 'rgba(88,166,255,.55)' : 'rgba(240,136,62,.55)';
        const dotStroke = isA ? '#58a6ff' : '#f0883e';
        html += '<circle cx="'+px.toFixed(1)+'" cy="'+py.toFixed(1)+
          '" r="'+rad.toFixed(1)+'" fill="'+dotFill+
          '" stroke="'+dotStroke+'" stroke-width=".5"/>';
      }
    }
  }

  svg.innerHTML = html;
}

/* ---- board rendering ---- */
function renderBoard() {
  const svg = document.getElementById('bsvg');
  const ph = document.getElementById('ph');
  const tl = document.getElementById('tl');
  const info = document.getElementById('info');
  const mvEl = document.getElementById('mv');

  if (!gameDetail || !gameDetail.turns || !gameDetail.turns.length) {
    svg.style.display='none'; ph.style.display=''; tl.style.display='none';
    info.innerHTML = ''; mvEl.innerHTML = '';
    document.getElementById('title').textContent = 'Select a game';
    return;
  }

  svg.style.display=''; ph.style.display='none'; tl.style.display='flex';

  const turns = gameDetail.turns;
  const slider = document.getElementById('tsl');
  slider.max = turns.length - 1;
  slider.value = curTurn;

  const t = turns[curTurn];
  drawHex(t.board, t.top_pairs, t.player);

  // title
  const val = gameDetail.value;
  const outcomeTag = gameDetail.drawn
    ? '<span class="tag tD">Draw</span>'
    : val > 0
      ? '<span class="tag tA">P1 wins</span>'
      : '<span class="tag tB">P2 wins</span>';
  document.getElementById('title').textContent = 'Game #' + gameDetail.game_id;

  // info bar
  const pTag = t.player === 1
    ? '<span class="tag tA">P1</span>'
    : '<span class="tag tB">P2</span>';
  info.innerHTML =
    '<span>Round <span class="v">' + gameDetail.round_id + '</span></span>' +
    '<span>Turn <span class="v">' + (curTurn+1) + ' / ' + turns.length + '</span></span>' +
    '<span>Moves <span class="v">' + t.move_count + '</span></span>' +
    '<span>Left <span class="v">' + t.moves_left + '</span></span>' +
    '<span>To play ' + pTag + '</span>' +
    '<span>Value <span class="v">' + (val>0?'+':'') + val.toFixed(1) + '</span></span> ' +
    outcomeTag;

  // timeline label
  document.getElementById('tlbl').textContent =
    'Turn ' + (curTurn+1) + ' / ' + turns.length +
    '  (move ' + t.move_count + ')';

  // top pairs
  if (t.top_pairs && t.top_pairs.length) {
    mvEl.innerHTML = t.top_pairs.map(p =>
      '<div class="mc">' +
      '<span class="cd">(' + p.q1 + ',' + p.r1 + ')-(' + p.q2 + ',' + p.r2 + ')</span>' +
      '<span class="vs">' + p.v + 'v</span>' +
      '</div>'
    ).join('');
  } else {
    mvEl.innerHTML = '';
  }
}

/* ---- timeline controls ---- */
function goTurn(t) {
  if (!gameDetail) return;
  curTurn = Math.max(0, Math.min(t, gameDetail.turns.length - 1));
  renderBoard();
}
document.getElementById('tb0').onclick = () => goTurn(0);
document.getElementById('tbp').onclick = () => goTurn(curTurn - 1);
document.getElementById('tbn').onclick = () => goTurn(curTurn + 1);
document.getElementById('tbe').onclick = () => { if(gameDetail) goTurn(gameDetail.turns.length-1); };
document.getElementById('tsl').oninput = e => goTurn(parseInt(e.target.value));

document.addEventListener('keydown', e => {
  if (!gameDetail || e.target.tagName === 'INPUT') return;
  switch (e.key) {
    case 'ArrowLeft':  e.preventDefault(); goTurn(curTurn-1); break;
    case 'ArrowRight': e.preventDefault(); goTurn(curTurn+1); break;
    case 'Home':       e.preventDefault(); goTurn(0); break;
    case 'End':        e.preventDefault(); if(gameDetail) goTurn(gameDetail.turns.length-1); break;
  }
});

/* ---- init ---- */
loadRounds();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
