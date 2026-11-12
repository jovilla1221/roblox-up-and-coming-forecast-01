// roblox-up-and-coming-forecast-01 entry: an occupancy forecaster for Roblox's
// Up-and-Coming chart (Node 22, standard library only, no network, deterministic).
//
// The scored object is ONE forecast graded against every hourly read of a 72-hour
// window, so the right target is not "the chart at one future moment" but each
// game's occupancy frequency at each place over the whole window. This program
// therefore:
//   1. reads the full history and, per game, estimates a latent expected log-rank
//      from an exponentially time-weighted average of its ranks (recent reads
//      weigh most), plus a damped momentum drift from a weighted regression of
//      log-rank on time and a correction from its playing-count trend;
//   2. turns that latent rank into a distribution over future places by Monte
//      Carlo: at each of 36 horizon buckets spanning the window it draws a
//      full ranking of the candidate games with fat-tailed noise whose scale
//      grows with the square root of the horizon (calibrated on the history's
//      own rank transitions), so uncertainty widens the further out the hour is;
//   3. averages the simulated occupancy over all buckets, weighting each
//      bucket down by that hour's unnamed share (small early, large late)
//      for games the forecast does not name (new and surprise entrants), and
//      caps any single game's probability at a place;
//   4. repeats the whole estimate at three half-lives (5h, 6h, 7.5h) and
//      averages the forecasts: the chart's memory length is itself
//      uncertain, and the pooled forecast scored better in backtests than
//      any single half-life.
//
// Games whose Roblox `created` date is known (the final history's recent reads
// carry details) are removed from buckets after they age off the chart at
// created + 28 days. All randomness comes from a fixed-seed PRNG, so two runs
// on the same history write byte-identical output.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
//
// The RF_* environment overrides exist only so the constants below can be tuned
// against backtests on the practice history; the grader sets none of them and
// the committed defaults are the tuned values.

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

function envNum(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  const v = Number(raw);
  return Number.isFinite(v) ? v : fallback;
}

/** Half-life (hours) of the exponential weights on past reads. Backtests on the
 *  practice history priced this short: the chart turns over fast, so the last
 *  day of reads matters far more than the week before it. */
const HALF_LIFE_H = envNum('RF_HL', 6);
/** Noise scale: latent-strength noise sd = SIGMA0 + SIGMA_RATE * sqrt(horizon h). */
const SIGMA0 = envNum('RF_SIGMA0', 0.1);
const SIGMA_RATE = envNum('RF_SIGMA', 0.11);
/** Momentum: the regression slope is scaled by this factor, then applied with a
 *  saturating horizon DAMP_TAU_H so a trend cannot run away over 72 hours.
 *  Backtests said momentum is real money (removing it cost ~2 nats), at about
 *  face value rather than shrunk. */
const VELOCITY_SHRINK = envNum('RF_VSHRINK', 1.1);
const DAMP_TAU_H = envNum('RF_TAU', 40);
const MAX_DRIFT = envNum('RF_MAXDRIFT', 1.6);
/** Weight of the playing-count log-slope in the drift (rank already reflects
 *  growth, so this is a correction on top of it, not the main signal). */
const PLAYING_WEIGHT = envNum('RF_WPLAY', 0.5);
/** Strength bonus for a good latest Top Trending place (log-rank units).
 *  Backtests priced this at zero, so it ships disabled; the parsing stays so
 *  the signal is one constant away if denser final history argues for it. */
const TREND_BONUS = envNum('RF_TREND', 0);
/** Fat tails: with probability JUMP_P a game's noise is scaled by JUMP_SCALE,
 *  which prices the chart's occasional 30-to-90-place leaps. */
const JUMP_P = envNum('RF_JUMPP', 0.13);
const JUMP_SCALE = envNum('RF_JUMPS', 2.1);
/** The unnamed share is not a constant: it is integrated per horizon bucket
 *  in the simulation (u(h) = 0.02 + 0.36 * (h/72)^0.85, averaging ~0.21),
 *  because the practice window's #1 was held for 3 of its 10 reads by a game
 *  absent from the whole history, and newcomers arrive late, not early. */
/** No game is given more than this at any one place; the practice window's
 *  closing #1 held #1 in none of its reads, so persistence is never priced
 *  anywhere near the baseline's 0.5. */
const P_CAP = envNum('RF_CAP', 0.3);
/** Rows below this probability are dropped (their mass joins the unnamed share). */
const P_MIN_ROW = 0.0004;
/** Monte Carlo draws per horizon bucket. */
const SIMS = Math.max(50, Math.floor(envNum('RF_SIMS', 900)));
/** Horizon buckets across the 72-hour window. */
const BUCKETS = 36;
/** Most candidate games carried into the simulation. */
const MAX_CANDIDATES = 72;
const DAY_MS = 86_400_000;
const HOUR_MS = 3_600_000;

/** Deterministic PRNG (mulberry32) so runs are byte-identical. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface ChartRow {
  rank: number;
  universe_id: number;
  playing?: number | null;
  up_votes?: number | null;
  down_votes?: number | null;
}

function median(xs: number[]): number {
  const a = [...xs].sort((x, y) => x - y);
  const m = Math.floor(a.length / 2);
  return a.length % 2 === 1 ? a[m]! : (a[m - 1]! + a[m]!) / 2;
}

interface Snapshot {
  atMs: number;
  chart: ChartRow[];
  trending: Map<number, number>;
  createdMs: Map<number, number>;
}

/** Pull every game's created timestamp out of one read's details, whatever
 *  shape the recorder used (map by universe id, or a list of detail rows). */
function extractCreated(details: unknown, out: Map<number, number>): void {
  if (details === null || details === undefined) return;
  const readOne = (uidRaw: unknown, value: unknown): void => {
    if (value === null || typeof value !== 'object') return;
    const rec = value as Record<string, unknown>;
    const created = rec['created'];
    if (typeof created !== 'string') return;
    const ms = Date.parse(created);
    if (!Number.isFinite(ms)) return;
    const uid = Number(
      (typeof rec['universe_id'] === 'number' ? rec['universe_id'] : undefined) ??
        (typeof rec['universeId'] === 'number' ? rec['universeId'] : undefined) ??
        (typeof rec['id'] === 'number' ? rec['id'] : undefined) ??
        uidRaw,
    );
    if (Number.isSafeInteger(uid) && uid > 0) out.set(uid, ms);
  };
  if (Array.isArray(details)) {
    for (const item of details) readOne(undefined, item);
  } else if (typeof details === 'object') {
    for (const [key, value] of Object.entries(details as Record<string, unknown>)) {
      readOne(key, value);
    }
  }
}

function loadSnapshots(): { snaps: Snapshot[]; places: number; output: string; windowFromMs: number } {
  const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
    places?: number;
    output?: string;
    window?: { from?: string };
    history_until?: string;
  };
  const text = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8');
  const snaps: Snapshot[] = [];
  for (const line of text.split('\n')) {
    if (line.trim() === '') continue;
    const raw = JSON.parse(line) as {
      at?: string;
      up_and_coming?: ChartRow[];
      top_trending?: ChartRow[] | null;
      details?: unknown;
    };
    const atMs = Date.parse(raw.at ?? '');
    if (!Number.isFinite(atMs) || !Array.isArray(raw.up_and_coming)) continue;
    const trending = new Map<number, number>();
    if (Array.isArray(raw.top_trending)) {
      for (const row of raw.top_trending) {
        if (typeof row?.universe_id === 'number' && typeof row?.rank === 'number') {
          trending.set(row.universe_id, row.rank);
        }
      }
    }
    const createdMs = new Map<number, number>();
    extractCreated(raw.details, createdMs);
    snaps.push({ atMs, chart: raw.up_and_coming, trending, createdMs });
  }
  snaps.sort((a, b) => a.atMs - b.atMs);
  const parsedWindow = Date.parse(task.window?.from ?? task.history_until ?? '');
  const windowFromMs = Number.isFinite(parsedWindow)
    ? parsedWindow
    : snaps.length
      ? snaps[snaps.length - 1]!.atMs
      : 0;
  return { snaps, places: task.places ?? 10, output: task.output ?? 'forecast.json', windowFromMs };
}

interface GameModel {
  uid: number;
  /** Residual log-playing vs the closing chart's rank expectation (OLS,
   *  clipped to +/-2): a big audience for its place marks a sticky hit. */
  zplay: number;
  /** Time-weighted mean log-rank at the end of history (lower is stronger). */
  mu0: number;
  /** Drift coefficient: total log-rank drift = driftCoef * DAMP_TAU_H *
   *  (1 - exp(-h / DAMP_TAU_H)), combining rank velocity and playing slope. */
  driftCoef: number;
  rankSlope: number;
  beta: number;
  /** Present since the first read and not audience-strong: a stale game
   *  whose occupancy weight decays over the window (tenure hazard). */
  oldFlat: boolean;
  /** One-directional vote-ratio penalty coefficient (per hour, applied over
   *  the first 24h, hard-capped in total at ~0.10 log-rank). Zero unless the
   *  closing read carries vote counts, which the practice archive's do not. */
  voteCoef: number;
  lastRank: number;
  ageOutMs: number | null;
}

/** Build one latent model per candidate game from the whole history. */
function buildModels(snaps: Snapshot[], halfLifeH: number): GameModel[] {
  const last = snaps[snaps.length - 1]!;
  const tEnd = last.atMs;
  interface Obs {
    tH: number;
    logRank: number;
    logPlaying: number;
    w: number;
    /** Same trapezoidal weight with a 12h half-life, used for the playing
     *  slope only when the recent history is dense (hourly reads). */
    wP: number;
  }
  const series = new Map<number, Obs[]>();
  const createdAll = new Map<number, number>();
  const gapsH: number[] = [];
  for (let gi = 1; gi < snaps.length; gi++) gapsH.push((snaps[gi]!.atMs - snaps[gi - 1]!.atMs) / HOUR_MS);
  const medianGapH = gapsH.length > 0 ? median(gapsH) : 1;
  const dtW: number[] = snaps.map((s, i) => {
    if (snaps.length < 2) return 1;
    if (i === 0) return Math.min(8, (snaps[1]!.atMs - s.atMs) / HOUR_MS);
    if (i === snaps.length - 1) return Math.max(medianGapH, Math.min(8, (s.atMs - snaps[i - 1]!.atMs) / HOUR_MS));
    return Math.min(8, (snaps[i + 1]!.atMs - snaps[i - 1]!.atMs) / (2 * HOUR_MS));
  });
  for (let si = 0; si < snaps.length; si++) {
    const snap = snaps[si]!;
    const ageH = (tEnd - snap.atMs) / HOUR_MS;
    const w = Math.pow(0.5, ageH / halfLifeH) * dtW[si]!;
    for (const row of snap.chart) {
      if (typeof row?.universe_id !== 'number' || typeof row?.rank !== 'number') continue;
      const playing = typeof row.playing === 'number' && row.playing > 0 ? row.playing : 1;
      let arr = series.get(row.universe_id);
      if (arr === undefined) {
        arr = [];
        series.set(row.universe_id, arr);
      }
      arr.push({
        tH: snap.atMs / HOUR_MS,
        logRank: Math.log(row.rank),
        logPlaying: Math.log(playing + 10),
        w,
        wP: dtW[si]! * Math.pow(0.5, ageH / 12),
      });
    }
    for (const [uid, ms] of snap.createdMs) createdAll.set(uid, ms);
  }
  const lastRank = new Map<number, number>();
  for (const row of last.chart) lastRank.set(row.universe_id, row.rank);
  // OLS of log(playing) on log(rank) over the closing chart (rank <= 50):
  // a game's residual says whether its audience is large for its place.
  // Suggested by an Argon High review and kept only because truncated-history
  // backtests scored it better (+0.3 nats) than the model without it.
  const zplay = new Map<number, number>();
  {
    const pts: Array<[number, number]> = [];
    for (const row of last.chart) {
      if (row.rank <= 50 && typeof row.playing === 'number' && row.playing > 0) {
        pts.push([Math.log(row.rank), Math.log(row.playing + 10)]);
      }
    }
    if (pts.length >= 10) {
      const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
      const my = pts.reduce((s, p) => s + p[1], 0) / pts.length;
      let sxx = 0;
      let sxy = 0;
      for (const [x, y] of pts) {
        sxx += (x - mx) * (x - mx);
        sxy += (x - mx) * (y - my);
      }
      const b = sxx > 1e-9 ? sxy / sxx : 0;
      const a = my - b * mx;
      let sr = 0;
      for (const [x, y] of pts) sr += (y - (a + b * x)) * (y - (a + b * x));
      const sd = Math.sqrt(sr / pts.length) || 1;
      for (const row of last.chart) {
        if (typeof row.playing === 'number' && row.playing > 0) {
          const z = (Math.log(row.playing + 10) - (a + b * Math.log(row.rank))) / sd;
          zplay.set(row.universe_id, Math.max(-2, Math.min(2, z)));
        }
      }
    }
  }
  // Dense-regime guard: with hourly reads in the last 48 hours, a 6h
  // half-life playing slope is mostly the daily cycle, so the playing
  // regression switches to the 12h half-life weights. Never triggers on
  // sparse histories like the practice archive.
  const densePlays = snaps.filter((sn) => sn.atMs > tEnd - 48 * HOUR_MS).length >= 30;
  // Vote-ratio penalty (protocol from an Argon High review): a robust
  // cross-sectional z-score of each game's smoothed like ratio on the
  // closing read; only the worst tail (z < -1.5) is penalised, by at most
  // ~0.10 log-rank in total. Needs at least 25 games with >= 100 votes;
  // archive reads carry no votes at all, which disables it completely.
  const voteCoef = new Map<number, number>();
  {
    const rows = last.chart.filter(
      (r) => typeof r.up_votes === 'number' && typeof r.down_votes === 'number' && r.up_votes + r.down_votes >= 100,
    );
    if (rows.length >= 25) {
      const medRaw = median(rows.map((r) => r.up_votes! / (r.up_votes! + r.down_votes!)));
      const smoothed = rows.map((r) => (r.up_votes! + 100 * medRaw) / (r.up_votes! + r.down_votes! + 100));
      const medS = median(smoothed);
      const mad = median(smoothed.map((x) => Math.abs(x - medS)));
      const scale = 1.4826 * mad + 0.02;
      rows.forEach((r, i) => {
        const z = (smoothed[i]! - medS) / scale;
        if (z < -1.5) voteCoef.set(r.universe_id, 0.0035 * Math.min(-z - 1.5, 1.2));
      });
    }
  }
  // Created-field validity guards: the age-out below trusts `created` as
  // the chart's 28-day clock, but that field never appears in the practice
  // archive, so the path is untested there. If games already older than
  // the limit are still on the closing chart, the field is evidently not
  // the chart's clock: exempt those games, and if they are common, switch
  // age-out off entirely rather than floor a live top game for 72 hours.
  let createdBroken = false;
  {
    let withCreated = 0;
    let old = 0;
    for (const row of last.chart) {
      const c = createdAll.get(row.universe_id);
      if (c === undefined) continue;
      withCreated++;
      if ((tEnd - c) / DAY_MS > 28.5) old++;
    }
    createdBroken = withCreated > 0 && old / withCreated > 0.15;
  }
  const recentBest = new Map<number, number>();
  for (const snap of snaps) {
    if ((tEnd - snap.atMs) / HOUR_MS > 96) continue;
    for (const row of snap.chart) {
      const prev = recentBest.get(row.universe_id);
      if (prev === undefined || row.rank < prev) recentBest.set(row.universe_id, row.rank);
    }
  }
  const models: GameModel[] = [];
  for (const [uid, obs] of series) {
    let sw = 0;
    let sLogRank = 0;
    let sT = 0;
    let sLogPlaying = 0;
    for (const o of obs) {
      sw += o.w;
      sLogRank += o.w * o.logRank;
      sT += o.w * o.tH;
      sLogPlaying += o.w * o.logPlaying;
    }
    if (sw <= 0) continue;
    const mu0raw = sLogRank / sw;
    const meanT = sT / sw;
    const meanP = sLogPlaying / sw;
    // Weighted regression slopes of log-rank and log-playing on time.
    let sTT = 0;
    let sTR = 0;
    let sTP = 0;
    for (const o of obs) {
      const dt = o.tH - meanT;
      sTT += o.w * dt * dt;
      sTR += o.w * dt * (o.logRank - mu0raw);
      sTP += o.w * dt * (o.logPlaying - meanP);
    }
    let rankSlope = sTR / (sTT + 10 * sw);
    let playSlope = sTP / (sTT + 10 * sw);
    if (densePlays) {
      // Recompute the playing slope with the 12h half-life weights so the
      // regression spans a full daily cycle instead of one diurnal swing.
      let swp = 0;
      let stp = 0;
      let spp = 0;
      for (const o of obs) {
        swp += o.wP;
        stp += o.wP * o.tH;
        spp += o.wP * o.logPlaying;
      }
      if (swp > 0) {
        const mT = stp / swp;
        const mP = spp / swp;
        let num = 0;
        let den = 0;
        for (const o of obs) {
          const d = o.tH - mT;
          num += o.wP * d * (o.logPlaying - mP);
          den += o.wP * d * d;
        }
        playSlope = num / (den + 10 * swp);
      }
    }
    rankSlope = Math.max(-0.05, Math.min(0.05, rankSlope));
    playSlope = Math.max(-0.06, Math.min(0.06, playSlope));
    // A game with too little history has no measurable trend; give it none.
    if (obs.length < 2) {
      rankSlope = 0;
      playSlope = 0;
    }
    const driftCoef = VELOCITY_SHRINK * rankSlope - PLAYING_WEIGHT * playSlope;
    // The latest Top Trending place is a mild independent strength signal.
    let mu0 = mu0raw;
    const trendRank = last.trending.get(uid);
    if (trendRank !== undefined && trendRank <= 40) {
      mu0 -= TREND_BONUS * (1 - trendRank / 41);
    }
    let betaUsed = 0;
    {
      const lr = lastRank.get(uid) ?? 999;
      if (lr < 999 && Math.log(lr) < mu0) {
        let fsH = Infinity;
        for (const o of obs) if (o.tH < fsH) fsH = o.tH;
        const isNew = fsH > snaps[0]!.atMs / HOUR_MS + 6;
        const ageSeen = tEnd / HOUR_MS - fsH;
        let beta = 0.2 * Math.max(0, (zplay.get(uid) ?? 0) - 0.5) + 0.2 * (mu0 - Math.log(lr) > 0.35 ? 1 : 0);
        if (isNew && ageSeen <= 30) beta += 0.3;
        beta = Math.min(0.6, Math.max(0, beta));
        betaUsed = beta;
        mu0 = (1 - beta) * mu0 + beta * Math.log(lr);
      }
    }
    const created = createdAll.get(uid);
    // A game older than 27.8 days that is still on the closing chart is
    // exempt from age-out on the same evidence as the global guard above.
    let ageOutMs: number | null = null;
    if (created !== undefined && !createdBroken && (tEnd - created) / DAY_MS <= 27.8) {
      ageOutMs = created + 28 * DAY_MS;
    }
    models.push({
      uid,
      zplay: zplay.get(uid) ?? 0,
      mu0,
      driftCoef,
      rankSlope,
      beta: betaUsed,
      oldFlat: Math.min(...obs.map((o) => o.tH)) <= snaps[0]!.atMs / HOUR_MS + 0.01 && (zplay.get(uid) ?? 0) < 0.5,
      voteCoef: voteCoef.get(uid) ?? 0,
      lastRank: lastRank.get(uid) ?? 999,
      ageOutMs,
    });
  }
  // Candidate set: anything that ends near the top, has recently been near the
  // top, or is climbing fast. Deep, flat, stale games reach the top 10 in 72
  // hours too rarely to be worth rows; the unnamed share covers them.
  const candidates = models.filter((m) => {
    if (m.lastRank <= 55) return true;
    if (Math.exp(m.mu0) <= 48) return true;
    if ((recentBest.get(m.uid) ?? 999) <= 20) return true;
    if (m.driftCoef < -0.012 && Math.exp(m.mu0) <= 80) return true;
    return false;
  });
  candidates.sort((a, b) => a.mu0 - b.mu0 || a.uid - b.uid);
  return candidates.slice(0, MAX_CANDIDATES);
}

function forecastFor(
  snaps: Snapshot[],
  places: number,
  windowFromMs: number,
  halfLifeH: number,
): { probabilities: Array<{ universe_id: number; place: number; p: number }>; scoredList: Array<{ uid: number; score: number; meanRank: number }> } {
  const last = snaps[snaps.length - 1]!;
  const models = buildModels(snaps, halfLifeH);
  if (models.length < places) throw new Error('not enough candidate games');

  const rand = mulberry32(0x5eed1234);
  // Box-Muller with a one-value cache, driven only by the seeded PRNG.
  let spare: number | null = null;
  const gauss = (): number => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    do {
      u = rand();
    } while (u <= 1e-12);
    const v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };

  const offsetH = Math.max(0, (windowFromMs - last.atMs) / HOUR_MS);
  const n = models.length;
  const occupancy: Float64Array[] = models.map(() => new Float64Array(places));
  const rankSum = new Float64Array(n);
  const base = new Float64Array(n);
  const noisy = new Float64Array(n);
  const active: number[] = [];
  let draws = 0;
  for (let b = 0; b < BUCKETS; b++) {
    const h = offsetH + 1 + 2 * b;
    const bucketMs = last.atMs + h * HOUR_MS;
    const sigma = Math.min(0.1, 0.045 + 0.035 * offsetH) + SIGMA_RATE * Math.sqrt(h);
    // The unnamed share grows over the window (few new entrants in the first
    // hours, many by hour 72); each bucket's occupancy is weighted by the
    // share of that hour still held by games the history knows. An Argon High
    // review proposed this curve; backtests scored it neutral-to-better.
    const uBucket = 0.02 + 0.36 * Math.pow(Math.min(h, 72) / 72, 0.85);
    const driftFactor = DAMP_TAU_H * (1 - Math.exp(-h / DAMP_TAU_H));
    active.length = 0;
    for (let i = 0; i < n; i++) {
      const m = models[i]!;
      if (m.ageOutMs !== null && bucketMs >= m.ageOutMs) continue;
      const drift = Math.max(-MAX_DRIFT, Math.min(MAX_DRIFT, m.driftCoef * driftFactor));
      // Two reviewed corrections ride on the momentum drift: a game whose
      // audience is large for its place (zplay > 0) is pulled up for the
      // first day, and a game already at the very top gets a gentle burnout
      // drift back down, because a growth chart's leader is there on a
      // growth spike that fades. Both survived backtest verification.
      base[i] = -(m.mu0 + drift - 0.01 * m.zplay * Math.min(h, 24) + m.voteCoef * Math.min(h, 24) + (m.mu0 <= Math.log(3) ? 0.012 * h : 0));
      active.push(i);
    }
    if (active.length === 0) continue;
    for (let s = 0; s < SIMS; s++) {
      for (const i of active) {
        let eps = gauss();
        if (rand() < JUMP_P) eps *= JUMP_SCALE;
        noisy[i] = base[i]! + sigma * Math.exp(-0.12 * models[i]!.zplay) * eps;
      }
      active.sort((x, y) => noisy[y]! - noisy[x]! || models[x]!.uid - models[y]!.uid);
      const top = Math.min(places, active.length);
      for (let k = 0; k < top; k++) {
        const tenureW = models[active[k]!]!.oldFlat ? 1 - 0.25 * Math.pow(Math.min(h, 72) / 72, 1.2) : 1;
        occupancy[active[k]!]![k]! += (1 - uBucket) * tenureW;
      }
      for (let k = 0; k < active.length; k++) {
        rankSum[active[k]!]! += Math.min(k + 1, 21);
      }
      draws++;
    }
  }
  if (draws === 0) throw new Error('no simulation draws');

  // Occupancy frequency -> probabilities: the bucket weighting already left
  // the time-varying unnamed share out, so frequencies are used as they are,
  // cap each game at each place, drop negligible rows.
  const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
  const placeRows: Array<Array<{ uid: number; p: number }>> = Array.from({ length: places }, () => []);
  for (let i = 0; i < n; i++) {
    const m = models[i]!;
    const freqs: number[] = [];
    let tot = 0;
    for (let k = 0; k < places; k++) {
      const f = occupancy[i]![k]! / draws;
      freqs.push(f);
      tot += f;
    }
    const elig = m.mu0 > Math.log(8) && (m.rankSlope < -0.008 || m.beta >= 0.25 || m.zplay >= 0.75);
    const mean = tot / places;
    for (let k = 0; k < places; k++) {
      let freq = freqs[k]!;
      if (elig) freq = 0.75 * freq + 0.25 * mean;
      if (freq <= 0) continue;
      placeRows[k]!.push({ uid: m.uid, p: freq });
    }
  }
  for (let k = 0; k < places; k++) {
    const rows = placeRows[k]!;
    let excess = 0;
    for (const r of rows) if (r.p > P_CAP) excess += r.p - P_CAP;
    if (excess > 0) {
      for (const r of rows) if (r.p > P_CAP) r.p = P_CAP;
      let give = excess * 0.75;
      for (let iter = 0; iter < 24 && give > 1e-12; iter++) {
        const below = rows.filter((r) => r.p < P_CAP - 1e-12);
        const tot = below.reduce((s, r) => s + r.p, 0);
        if (below.length === 0 || tot <= 0) break;
        let used = 0;
        for (const r of below) {
          const add = Math.min(P_CAP - r.p, (give * r.p) / tot);
          r.p += add;
          used += add;
        }
        give -= used;
        if (used <= 1e-12) break;
      }
    }
    for (const r of rows) r.p = Math.round(r.p * 1e6) / 1e6;
    for (let i = rows.length - 1; i >= 0; i--) if (rows[i]!.p < P_MIN_ROW) rows.splice(i, 1);
  }
  for (let k = 0; k < places; k++) {
    const rows = placeRows[k]!;
    rows.sort((a, b) => b.p - a.p || a.uid - b.uid);
    for (const row of rows.slice(0, 90)) {
      probabilities.push({ universe_id: row.uid, place: k + 1, p: row.p });
    }
  }

  // Top 10 guess: rank games by an occupancy-weighted score (a game likely to
  // sit at place 1 outranks one equally likely to sit at place 10), breaking
  // ties by the lower simulated mean rank, then by universe id.
  const scored = models.map((m, i) => {
    let score = 0;
    let mass = 0;
    for (let k = 0; k < places; k++) {
      const f = occupancy[i]![k]! / draws;
      score += f * (places - k);
      mass += f;
    }
    return { uid: m.uid, score, mass, meanRank: rankSum[i]! / draws };
  });
  scored.sort((a, b) => b.score - a.score || a.meanRank - b.meanRank || a.uid - b.uid);
  return { probabilities, scoredList: scored };
}

function main(): void {
  const { snaps, places, output, windowFromMs } = loadSnapshots();
  if (snaps.length === 0) throw new Error('empty history');
  const last = snaps[snaps.length - 1]!;
  // Linear pool over three half-lives (5h, 6h, 7.5h at the default): the
  // chart's memory length is itself uncertain, and averaging the three
  // forecasts scored better in backtests than any single half-life.
  const poolHLs = [HALF_LIFE_H * (5 / 6), HALF_LIFE_H, HALF_LIFE_H * 1.25];
  const acc = new Map<string, { universe_id: number; place: number; p: number }>();
  const scoreAcc = new Map<number, { score: number; meanRank: number }>();
  for (const hl of poolHLs) {
    const f = forecastFor(snaps, places, windowFromMs, hl);
    for (const row of f.probabilities) {
      const key = `${row.universe_id}|${row.place}`;
      const cur = acc.get(key);
      if (cur) cur.p += row.p / poolHLs.length;
      else acc.set(key, { universe_id: row.universe_id, place: row.place, p: row.p / poolHLs.length });
    }
    for (const sc of f.scoredList) {
      const cur = scoreAcc.get(sc.uid);
      if (cur) {
        cur.score += sc.score / poolHLs.length;
        cur.meanRank += sc.meanRank / poolHLs.length;
      } else {
        scoreAcc.set(sc.uid, { score: sc.score / poolHLs.length, meanRank: sc.meanRank / poolHLs.length });
      }
    }
  }
  const byPlace = new Map<number, Array<{ universe_id: number; place: number; p: number }>>();
  for (const row of acc.values()) {
    const arr = byPlace.get(row.place) ?? [];
    arr.push({ universe_id: row.universe_id, place: row.place, p: Math.round(row.p * 1e6) / 1e6 });
    byPlace.set(row.place, arr);
  }
  const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
  for (let k = 1; k <= places; k++) {
    const arr = (byPlace.get(k) ?? []).sort((a, b) => b.p - a.p || a.universe_id - b.universe_id);
    for (const row of arr.slice(0, 90)) probabilities.push(row);
  }
  const pooled = [...scoreAcc.entries()].map(([uid, v]) => ({ uid, score: v.score, meanRank: v.meanRank }));
  pooled.sort((a, b) => b.score - a.score || a.meanRank - b.meanRank || a.uid - b.uid);
  const top10 = pooled.slice(0, places).map((x) => x.uid);
  if (top10.length < places) {
    // Defensive: pad from the closing chart (never expected in practice).
    for (const row of [...last.chart].sort((a, b) => a.rank - b.rank)) {
      if (top10.length >= places) break;
      if (!top10.includes(row.universe_id)) top10.push(row.universe_id);
    }
  }

  writeFileSync(join(OUTPUT_DIR, output), `${JSON.stringify({ probabilities, top10 })}\n`);
}


main();
