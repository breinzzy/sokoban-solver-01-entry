// Sokoban solver for make.fun's sokoban-solver-01-run-20261121.
//
// Strategy (see the kit's RULES.txt: every level is a 1-wide hallway with
// 1-4 rooms off it, each room 3-4 boxes in a 6x6 slot, one door, and the
// level optimum P* is the sum of the rooms' optima):
//   1. Split the level into rooms (flood fill after cutting the hallway's
//      dead squares) and solve each room on its own, the other rooms'
//      boxes standing as walls. The player walks the hallway between rooms.
//   2. Each room is solved by A* over pushes (never single steps): a state
//      is the sorted box layout plus the player's reachable region, named
//      by its lowest square. Cost is pushes; walks between pushes are free.
//   3. Heuristic: the minimum-cost matching between boxes and goals over
//      push-distance tables (one reverse push-BFS per goal), which dominates
//      the Manhattan sum and stays admissible. A state with no finite
//      perfect matching is dead.
//   4. Deadlock pruning: dead squares from a reverse pull/push BFS from
//      the goals, plus frozen 2x2 blocks of walls and boxes. A push that
//      lands a box on a dead square (including the hallway) is pruned.
//   5. If A* does not prove a room optimal inside its work budget, a
//      greedy best-first fallback returns the first valid solution it
//      finds, so a hard room still scores instead of timing the set out.
//   6. Everything is budgeted by work (expanded states), never by the
//      wall clock, and there is no randomness: two runs write the same
//      bytes. The concatenated moves are replayed against the full level
//      before anything is written out.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// [dx, dy, walk letter, push letter], in the kit's order.
const DIRS: ReadonlyArray<readonly [number, number, string, string]> = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// Work budgets (expanded states), never the clock: deterministic across runs.
const ASTAR_BUDGET = 150_000;
const GREEDY_BUDGET = 150_000;
const WHOLE_ASTAR_BUDGET = 200_000;
const WHOLE_GREEDY_BUDGET = 200_000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

interface Solved {
  pushes: number;
  moves: string;
  end: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which a lone box can be pushed onto some goal: reverse
// push-BFS from the goals. A box at y can be pushed to x = y + d when the
// player can stand at y - d, so y is live when x is live and both y and
// y - d are open floor.
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Push distances: for each goal, the fewest pushes moving a lone box from
// every square onto that goal (-1 when impossible). Same reverse BFS as
// liveSquares, recording the distance.
function pushDistances(level: Level): Int32Array[] {
  const { w, wall, goals } = level;
  return goals.map((g) => {
    const dist = new Int32Array(wall.length).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const y = x - d;
        if (!wall[y] && !wall[y - d] && dist[y] < 0) {
          dist[y] = dist[x]! + 1;
          queue.push(y);
        }
      }
    }
    return dist;
  });
}

// Manhattan distance from every square to its nearest goal.
function manhattanTable(level: Level): Int32Array {
  const { w, wall, goals } = level;
  const dist = new Int32Array(wall.length).fill(1 << 30);
  const queue = [...goals];
  for (const g of goals) dist[g] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const q = x + dy * w + dx;
      if (!wall[q] && dist[q] > dist[x]! + 1) {
        dist[q] = dist[x]! + 1;
        queue.push(q);
      }
    }
  }
  return dist;
}

// The level's rooms: cut the hallway row's dead squares out of the floor
// and flood fill what is left. Each part holding boxes is a room. Null
// when the level is no set of rooms (a box in the hallway, a goal outside
// every part, or a part whose boxes and goals do not match).
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  }
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Whether the box just pushed to `t` sits in a frozen 2x2 block of walls
// and boxes with a box off its goal.
function frozenBlock(
  level: Level,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a])) continue;
    if (!(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c])) continue;
    if (!(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) {
      return true;
    }
  }
  return false;
}

// The lowest square of the region the player reaches from `from` with
// `occupied` in the way, marking it in `seen` under `stamp`.
function reachLow(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// The shortest walk from `from` to `to` around walls and boxes, in lower
// LURD letters (u first on ties, as in the kit's reference solver).
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how: string[] = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

interface HeapEntry {
  f: number;
  h: number;
  seq: number;
  s: number;
}

function heapLess(a: HeapEntry, b: HeapEntry): boolean {
  return a.f < b.f || (a.f === b.f && (a.h < b.h || (a.h === b.h && a.seq < b.seq)));
}

function heapPush(heap: HeapEntry[], e: HeapEntry): void {
  let i = heap.length;
  heap.push(e);
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (!heapLess(heap[i]!, heap[p]!)) break;
    const tmp = heap[i]!;
    heap[i] = heap[p]!;
    heap[p] = tmp;
    i = p;
  }
}

function heapPop(heap: HeapEntry[]): HeapEntry {
  const top = heap[0]!;
  const last = heap.pop()!;
  if (heap.length > 0) {
    heap[0] = last;
    for (let i = 0; ; ) {
      let c = 2 * i + 1;
      if (c >= heap.length) break;
      if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
      if (!heapLess(heap[c]!, heap[i]!)) break;
      const tmp = heap[i]!;
      heap[i] = heap[c]!;
      heap[c] = tmp;
      i = c;
    }
  }
  return top;
}

// A* (greedy=false) or greedy best-first (greedy=true) over pushes for one
// level (a room sublevel, or the whole level as a fallback). Returns the
// push-optimal solution in A* mode, or the first valid solution in greedy
// mode, or null when the work budget runs out or no solution exists.
function solvePushes(
  sub: Level,
  maxExpand: number,
  greedy: boolean,
  useMatching: boolean,
): Solved | null {
  const { w, wall, goals } = sub;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(sub);
  if (sub.boxes.some((b) => !live[b])) return null;
  const dists = useMatching ? pushDistances(sub) : [];
  const man = useMatching ? null : manhattanTable(sub);

  // Admissible heuristic in pushes: the minimum-cost perfect matching
  // between boxes and goals over the push-distance tables (brute force
  // over the at most 4! assignments), or the Manhattan sum for big levels.
  const heuristic = (boxes: number[]): number => {
    if (useMatching) {
      const n = boxes.length;
      const used = new Array(n).fill(false);
      let best = Infinity;
      const rec = (i: number, sum: number): void => {
        if (sum >= best) return;
        if (i === n) {
          best = sum;
          return;
        }
        for (let j = 0; j < n; j++) {
          if (used[j]) continue;
          const d = dists[j]![boxes[i]!]!;
          if (d < 0) continue;
          used[j] = true;
          rec(i + 1, sum + d);
          used[j] = false;
        }
      };
      rec(0, 0);
      return best;
    }
    let s = 0;
    for (const b of boxes) s += man![b]!;
    return s;
  };

  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  const keyOf = (boxes: number[], low: number): string => low + ':' + boxes.join(',');

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const sBoxes: number[][] = [sub.boxes.slice()];
  const sPlayer: number[] = [sub.player];
  const sLow: number[] = [];
  const sG: number[] = [0];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const pushDir: number[] = [-1];
  const best = new Map<string, number>();

  const h0 = heuristic(sub.boxes);
  if (!Number.isFinite(h0)) return null;
  for (const b of sub.boxes) occupied[b] = 1;
  const low0 = reachLow(sub, occupied, sub.player, seen, ++stamp, stack);
  for (const b of sub.boxes) occupied[b] = 0;
  sLow.push(low0);
  best.set(keyOf(sub.boxes, low0), 0);
  let seq = 0;
  const heap: HeapEntry[] = [];
  heapPush(heap, { f: h0, h: h0, seq: seq++, s: 0 });

  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const e = heapPop(heap);
    const s = e.s;
    const boxes = sBoxes[s]!;
    // Stale heap entry: a cheaper copy of this state was recorded after
    // this one was pushed. Skip it without spending the work budget.
    if (best.get(keyOf(boxes, sLow[s]!)) !== sG[s]) continue;
    if (expanded >= maxExpand) return null;
    expanded++;
    if (solved(boxes)) {
      found = s;
      break;
    }
    for (const b of boxes) occupied[b] = 1;
    const here = ++stamp;
    reachLow(sub, occupied, sPlayer[s]!, seen, here, stack);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di]!;
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here) continue;
        if (wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(sub, occupied, goal, t);
        let h2 = Infinity;
        let low2 = -1;
        let key2 = '';
        if (!frozen) {
          const next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          h2 = heuristic(next);
          if (Number.isFinite(h2)) {
            low2 = reachLow(sub, occupied, b, mark, ++stamp, stack);
            key2 = keyOf(next, low2);
            const prevBest = best.get(key2);
            if (prevBest === undefined || (greedy ? false : prevBest > sG[s]! + 1)) {
              best.set(key2, sG[s]! + 1);
              const ns = sBoxes.length;
              sBoxes.push(next);
              sPlayer.push(b);
              sLow.push(low2);
              sG.push(sG[s]! + 1);
              parent.push(s);
              pushFrom.push(b);
              pushTo.push(t);
              pushDir.push(di);
              heapPush(heap, {
                f: greedy ? h2 : sG[s]! + 1 + h2,
                h: h2,
                seq: seq++,
                s: ns,
              });
            }
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;
  // Rebuild the push sequence, then expand each push into walk + push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = sub.player;
  let boxes = sub.boxes.slice();
  for (const s of path) {
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const di = pushDir[s]!;
    const [dx, dy, , pushLetter] = DIRS[di]!;
    const d = dy * w + dx;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(sub, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += pushLetter;
    player = from;
    const at = boxes.indexOf(from);
    boxes[at] = to;
    boxes.sort((a, b) => a - b);
  }
  return { pushes: path.length, moves, end: player };
}

// Whole-level fallback: the same push search on the full level.
function solveWhole(level: Level): string | null {
  const useMatching = level.boxes.length <= 6;
  const res =
    solvePushes(level, WHOLE_ASTAR_BUDGET, false, useMatching) ??
    solvePushes(level, WHOLE_GREEDY_BUDGET, true, useMatching);
  return res === null ? null : res.moves;
}

// Replay moves on the level; true when every box ends on a goal.
function verify(level: Level, moves: string): boolean {
  const { w, wall, goals } = level;
  const goal = new Uint8Array(wall.length);
  for (const g of goals) goal[g] = 1;
  const box = new Uint8Array(wall.length);
  for (const b of level.boxes) box[b] = 1;
  let player = level.player;
  for (const ch of moves) {
    const lower = ch.toLowerCase();
    const di = lower === 'u' ? 0 : lower === 'd' ? 1 : lower === 'l' ? 2 : 3;
    const [dx, dy] = DIRS[di]!;
    const d = dy * w + dx;
    const next = player + d;
    if (wall[next]) return false;
    if (box[next]) {
      const beyond = next + d;
      if (wall[beyond] || box[beyond]) return false;
      box[next] = 0;
      box[beyond] = 1;
    }
    player = next;
  }
  for (let i = 0; i < box.length; i++) {
    if (box[i] && !goal[i]) return false;
  }
  return true;
}

function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let moves = '';
    let player = level.player;
    for (const room of rooms) {
      const wall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub: Level = {
        w: level.w,
        h: level.h,
        wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      const res =
        solvePushes(sub, ASTAR_BUDGET, false, true) ??
        solvePushes(sub, GREEDY_BUDGET, true, true);
      if (res === null) return solveWhole(level) ?? '';
      moves += res.moves;
      player = res.end;
      room.done = true;
    }
    if (verify(level, moves)) return moves;
    return solveWhole(level) ?? '';
  }
  return solveWhole(level) ?? '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
