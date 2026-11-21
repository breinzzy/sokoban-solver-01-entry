// Push-optimal Sokoban solver: split the level into rooms, solve each room
// with A* over pushes (Hungarian matching over push-distance tables, dead
// squares + frozen 2x2 pruning), walk the hallway between rooms. Budgets
// are work (expanded states), never the clock: two runs write same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const D = [[0, -1, 'u', 'U'], [0, 1, 'd', 'D'], [-1, 0, 'l', 'L'], [1, 0, 'r', 'R']];
const AB = 150000, GB = 150000, WB = 200000;

function parse(t) {
  const rows = t.split('\n').filter(r => r.length > 0);
  const h = rows.length, w = Math.max(...rows.map(r => r.length));
  const wall = new Uint8Array(w * h).fill(1), goals = [], boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x], i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if ('.*+'.includes(ch)) goals.push(i);
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Push distances: for each goal, fewest pushes moving a lone box from every
// square onto it (-1 when impossible). Reverse push-BFS: a box at y can be
// pushed to x = y + d when the player can stand at y - d. A square is live
// when some goal reaches it, which is also the dead-square test.
function pdists(lv) {
  const { w, wall, goals } = lv;
  return goals.map(g => {
    const dist = new Int32Array(wall.length).fill(-1);
    dist[g] = 0;
    const q = [g];
    for (let qi = 0; qi < q.length; qi++) {
      const x = q[qi];
      for (const [dx, dy] of D) {
        const d = dy * w + dx, y = x - d;
        if (!wall[y] && !wall[y - d] && dist[y] < 0) { dist[y] = dist[x] + 1; q.push(y); }
      }
    }
    return dist;
  });
}

function liveOf(lv, ds) {
  const live = new Uint8Array(lv.wall.length);
  for (const d of ds) for (let i = 0; i < d.length; i++) if (d[i] >= 0) live[i] = 1;
  return live;
}

// The level's rooms: cut the hallway row's dead squares out of the floor
// and flood fill what is left. Null when the level is no set of rooms.
function rooms(lv) {
  const { w, wall, goals, boxes, player } = lv;
  const live = liveOf(lv, pdists(lv));
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some(b => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1), rs = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const sq = [s];
    part[s] = rs.length;
    for (let i = 0; i < sq.length; i++) for (const [dx, dy] of D) {
      const q = sq[i] + dy * w + dx;
      if (!wall[q] && !cut[q] && part[q] < 0) { part[q] = rs.length; sq.push(q); }
    }
    rs.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rs[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rs[part[g]].goals.push(g);
  if (goals.some(g => part[g] < 0)) return null;
  if (rs.some(r => r.boxes.length !== r.goals.length)) return null;
  return rs.filter(r => r.boxes.length > 0);
}

// Whether the box just pushed to t sits in a frozen 2x2 block of walls
// and boxes with a box off its goal.
function frozen(lv, occ, goal, t) {
  const { w, wall } = lv;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1, c = a + w, d = c + 1;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b])) continue;
    if (!(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goal[a]) || (occ[b] && !goal[b]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
  }
  return false;
}

// Lowest square of the region the player reaches from `from` with `occ`
// in the way, marking it in `seen` under `stamp`.
function reach(lv, occ, from, seen, stamp, stack) {
  const { w, wall } = lv;
  let low = from, n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occ[q] === 0 && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    }
  }
  return low;
}

// Shortest walk from `from` to `to` around walls and boxes, lower LURD.
function walk(lv, occ, from, to) {
  const { w, wall } = lv;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1), how = new Array(wall.length), q = [from];
  prev[from] = from;
  for (let qi = 0; qi < q.length; qi++) {
    const p = q[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of D) {
      const nq = p + dy * w + dx;
      if (wall[nq] || occ[nq] || prev[nq] !== -1) continue;
      prev[nq] = p; how[nq] = letter; q.push(nq);
    }
  }
  if (prev[to] === -1) throw new Error('walk: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

function hLess(a, b) { return a.f < b.f || (a.f === b.f && (a.h < b.h || (a.h === b.h && a.seq < b.seq))); }
function hPush(h, e) {
  let i = h.length;
  h.push(e);
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (!hLess(h[i], h[p])) break;
    const t = h[i]; h[i] = h[p]; h[p] = t; i = p;
  }
}
function hPop(h) {
  const top = h[0], last = h.pop();
  if (h.length > 0) {
    h[0] = last;
    for (let i = 0; ;) {
      let c = 2 * i + 1;
      if (c >= h.length) break;
      if (c + 1 < h.length && hLess(h[c + 1], h[c])) c++;
      if (!hLess(h[c], h[i])) break;
      const t = h[i]; h[i] = h[c]; h[c] = t; i = c;
    }
  }
  return top;
}

// A* (greedy=false) or greedy best-first (greedy=true) over pushes for one
// level. A* returns the push-optimal solution; greedy the first valid one;
// null when the work budget runs out or no solution exists.
function solve(sub, maxExpand, greedy, useMatching) {
  const { w, wall, goals } = sub, size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const ds = pdists(sub), live = liveOf(sub, ds);
  if (sub.boxes.some(b => !live[b])) return null;
  // Admissible heuristic in pushes: minimum-cost matching between boxes
  // and goals over the push-distance tables (at most 4! assignments), or
  // the per-box minimum sum for big levels. No finite matching = dead.
  const heur = boxes => {
    if (useMatching) {
      const n = boxes.length, used = new Array(n).fill(false);
      let best = Infinity;
      const rec = (i, sum) => {
        if (sum >= best) return;
        if (i === n) { best = sum; return; }
        for (let j = 0; j < n; j++) {
          if (used[j]) continue;
          const d = ds[j][boxes[i]];
          if (d < 0) continue;
          used[j] = true; rec(i + 1, sum + d); used[j] = false;
        }
      };
      rec(0, 0);
      return best;
    }
    let s = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (const d of ds) if (d[b] >= 0 && d[b] < m) m = d[b];
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  };
  const solved = boxes => boxes.every(b => goal[b]);
  const keyOf = (boxes, low) => low + ':' + boxes.join(',');
  const occ = new Uint8Array(size), seen = new Int32Array(size), mark = new Int32Array(size), stack = new Int32Array(size);
  let stamp = 0;
  const sBoxes = [sub.boxes.slice()], sPlayer = [sub.player], sLow = [], sG = [0];
  const parent = [-1], pFrom = [-1], pTo = [-1], pDir = [-1];
  const best = new Map();
  const h0 = heur(sub.boxes);
  if (!Number.isFinite(h0)) return null;
  for (const b of sub.boxes) occ[b] = 1;
  const low0 = reach(sub, occ, sub.player, seen, ++stamp, stack);
  for (const b of sub.boxes) occ[b] = 0;
  sLow.push(low0);
  best.set(keyOf(sub.boxes, low0), 0);
  let seq = 0;
  const heap = [];
  hPush(heap, { f: h0, h: h0, seq: seq++, s: 0 });
  let expanded = 0, found = -1;
  while (heap.length > 0) {
    const e = hPop(heap), s = e.s, boxes = sBoxes[s];
    if (best.get(keyOf(boxes, sLow[s])) !== sG[s]) continue; // stale entry
    if (expanded >= maxExpand) return null;
    expanded++;
    if (solved(boxes)) { found = s; break; }
    for (const b of boxes) occ[b] = 1;
    const here = ++stamp;
    reach(sub, occ, sPlayer[s], seen, here, stack);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = D[di], d = dy * w + dx, t = b + d;
        if (seen[b - d] !== here) continue;
        if (wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0; occ[t] = 1;
        const fr = frozen(sub, occ, goal, t);
        let h2 = Infinity, low2 = -1, key2 = '';
        if (!fr) {
          const next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          h2 = heur(next);
          if (Number.isFinite(h2)) {
            low2 = reach(sub, occ, b, mark, ++stamp, stack);
            key2 = keyOf(next, low2);
            const prev = best.get(key2);
            if (prev === undefined || (!greedy && prev > sG[s] + 1)) {
              best.set(key2, sG[s] + 1);
              const ns = sBoxes.length;
              sBoxes.push(next); sPlayer.push(b); sLow.push(low2); sG.push(sG[s] + 1);
              parent.push(s); pFrom.push(b); pTo.push(t); pDir.push(di);
              hPush(heap, { f: greedy ? h2 : sG[s] + 1 + h2, h: h2, seq: seq++, s: ns });
            }
          }
        }
        occ[t] = 0; occ[b] = 1;
      }
    }
    for (const b of boxes) occ[b] = 0;
  }
  if (found < 0) return null;
  const path = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '', player = sub.player, boxes = sub.boxes.slice();
  for (const s of path) {
    const from = pFrom[s], to = pTo[s], di = pDir[s];
    const [dx, dy, , pushLetter] = D[di], d = dy * w + dx;
    for (const b of boxes) occ[b] = 1;
    moves += walk(sub, occ, player, from - d);
    for (const b of boxes) occ[b] = 0;
    moves += pushLetter;
    player = from;
    boxes[boxes.indexOf(from)] = to;
    boxes.sort((a, b) => a - b);
  }
  return { pushes: path.length, moves, end: player };
}

function whole(lv) {
  const m = lv.boxes.length <= 6;
  const r = solve(lv, WB, false, m) ?? solve(lv, WB, true, m);
  return r === null ? null : r.moves;
}

// Replay moves on the level; true when every box ends on a goal.
function ok(lv, moves) {
  const { w, wall, goals } = lv, goal = new Uint8Array(wall.length), box = new Uint8Array(wall.length);
  for (const g of goals) goal[g] = 1;
  for (const b of lv.boxes) box[b] = 1;
  let player = lv.player;
  for (const ch of moves) {
    const lo = ch.toLowerCase(), di = lo === 'u' ? 0 : lo === 'd' ? 1 : lo === 'l' ? 2 : 3;
    const [dx, dy] = D[di], d = dy * w + dx, next = player + d;
    if (wall[next]) return false;
    if (box[next]) {
      const beyond = next + d;
      if (wall[beyond] || box[beyond]) return false;
      box[next] = 0; box[beyond] = 1;
    }
    player = next;
  }
  for (let i = 0; i < box.length; i++) if (box[i] && !goal[i]) return false;
  return true;
}

function solveLevel(lv) {
  const rs = rooms(lv);
  if (rs !== null) {
    let moves = '', player = lv.player;
    for (const room of rs) {
      const wall = lv.wall.slice();
      for (const other of rs) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub = { w: lv.w, h: lv.h, wall, goals: room.goals.slice().sort((a, b) => a - b), boxes: room.boxes.slice().sort((a, b) => a - b), player };
      const res = solve(sub, AB, false, true) ?? solve(sub, GB, true, true);
      if (res === null) return whole(lv) ?? '';
      moves += res.moves;
      player = res.end;
      room.done = true;
    }
    if (ok(lv, moves)) return moves;
    return whole(lv) ?? '';
  }
  return whole(lv) ?? '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(parse(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
