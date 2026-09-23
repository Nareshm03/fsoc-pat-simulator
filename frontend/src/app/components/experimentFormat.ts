/**
 * Pure experiment formatting + graph/compare data builders.
 *
 * No React, no DOM, no network — fully unit-testable. Components must
 * render these values verbatim (metrics are never recalculated here).
 */

export interface FrameRecord {
  detected: boolean;
  confidence: number;
  error_px: number;
  pat_state: string | null;
  time: number | null;
}

export interface MetricsLike {
  detection_rate: number;
  avg_confidence: number;
  initial_error: number | null;
  final_error: number | null;
  minimum_error: number | null;
  min_error: number | null;
  convergence_frame: number | null;
  convergence_time_s: number | null;
  locked_achieved: boolean;
  lock_retention_s: number;
  lost_count: number;
}

export interface ExperimentLike {
  experiment_id: string;
  detector: string;
  scenario: string | { name?: string };
  dt: number;
  metrics: MetricsLike;
  records?: FrameRecord[] | null;
}

/** Minimal shape the convergence graph needs (no metrics). */
export interface GraphExperiment {
  experiment_id: string;
  detector: string;
  scenario: string | { name?: string };
  dt: number;
  records?: FrameRecord[] | null;
}

/** Series palette: accent first, warning second (DESIGN.md semantic ramp). */
export const GRAPH_COLORS = ['#007aff', '#ff9f0a'];
export const MAX_COMPARE_SERIES = 2;

/** Client warmup budget: cold YOLO model load is seconds (backend perf
 * tests measure ~5 s first-tick cost), plus ~0.1-0.3 s async pipeline
 * fill. After this without any nominal detection, missing detections
 * read as genuine loss instead of warmup. */
export const YOLO_WARMUP_TIMEOUT_MS = 15000;
/** Transient "nominal detections started" confirmation dwell. */
export const YOLO_ACTIVE_FLASH_MS = 6000;

export type YoloPipelinePhase = 'inactive' | 'warming' | 'settled';

export interface YoloPipelineInput {
  selectedDetector: 'classical' | 'yolo';
  /** Backend-confirmed active detector (telemetry/ack), not the optimistic UI selection. */
  activeDetector: 'classical' | 'yolo';
  /** Control loop ticking with an open socket (running && !paused && connected). */
  loopLive: boolean;
  /** Any backend-confirmed YOLO detection with confidence > 0 since selection. */
  seenYoloDetection: boolean;
  /** Client warmup budget elapsed. */
  warmupExpired: boolean;
}

/**
 * Resolve the YOLO pipeline phase for warmup UX. Warming requires every
 * signal at once (selected + backend-confirmed + live + unseen +
 * unexpired), so a missing detection during warmup never renders as
 * beacon loss, and any settled state (first detection seen, or budget
 * elapsed) lets PAT SEARCH read as genuine loss.
 */
export function resolveYoloPipelinePhase(
  input: YoloPipelineInput
): YoloPipelinePhase {
  if (
    input.selectedDetector !== 'yolo' ||
    input.activeDetector !== 'yolo' ||
    !input.loopLive
  ) {
    return 'inactive';
  }
  if (!input.seenYoloDetection && !input.warmupExpired) {
    return 'warming';
  }
  return 'settled';
}

export type PatTone =
  | 'locked'
  | 'fine_track'
  | 'acquire'
  | 'search'
  | 'reacquire'
  | 'idle';

/**
 * Map a backend PAT state to a badge tone class suffix; anything unknown
 * (including nullish) falls back to idle. Case-insensitive so backend
 * casing changes cannot break badge styling.
 */
export function patStateTone(state: unknown): PatTone {
  const normalized =
    typeof state === 'string' ? state.toLowerCase() : '';
  if (normalized === 'locked') return 'locked';
  if (normalized === 'fine_track') return 'fine_track';
  if (normalized === 'acquire') return 'acquire';
  if (normalized === 'search') return 'search';
  if (normalized === 'reacquire') return 'reacquire';
  return 'idle';
}

export type DetectionRateTone = 'excellent' | 'good' | 'fair' | 'poor';

/**
 * Grade a 0-100 detection-rate percentage on the lock-quality scale.
 * Non-finite wire values grade poor instead of throwing.
 */
export function detectionRateTone(ratePct: number): DetectionRateTone {
  if (!Number.isFinite(ratePct)) return 'poor';
  if (ratePct >= 95) return 'excellent';
  if (ratePct >= 85) return 'good';
  if (ratePct >= 70) return 'fair';
  return 'poor';
}

export type LinkTone = 'nominal' | 'marginal' | 'outage' | 'nodata';

/** Map a backend link_state to a display tone; anything unknown => nodata. */
export function linkStateTone(state: unknown): LinkTone {
  if (state === 'NOMINAL') return 'nominal';
  if (state === 'MARGINAL') return 'marginal';
  if (state === 'OUTAGE') return 'outage';
  return 'nodata';
}

/**
 * Replace bare Infinity/-Infinity/NaN tokens outside JSON strings with null.
 * The Python backend emits them for out-of-range floats (e.g. unbounded
 * pointing loss); strict JSON.parse rejects them, which would drop every
 * telemetry message. String contents (incl. base64 images) pass through
 * untouched, so wire values are otherwise preserved verbatim.
 */
export function sanitizeNonJsonNumbers(text: string): string {
  let out = '';
  let inString = false;
  let escaped = false;
  let i = 0;
  const isIdentChar = (ch: string | undefined): boolean =>
    ch !== undefined && /[A-Za-z0-9_$]/.test(ch);
  while (i < text.length) {
    const ch = text[i];
    if (inString) {
      out += ch;
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      i += 1;
      continue;
    }
    if (ch === '"') {
      inString = true;
      out += ch;
      i += 1;
      continue;
    }
    let word: string | null = null;
    for (const w of ['-Infinity', 'Infinity', 'NaN']) {
      if (text.startsWith(w, i)) {
        word = w;
        break;
      }
    }
    if (
      word !== null &&
      !isIdentChar(text[i + word.length]) &&
      !isIdentChar(text[i - 1])
    ) {
      out += 'null';
      i += word.length;
      continue;
    }
    out += ch;
    i += 1;
  }
  return out;
}

/** Camera geometry mirrored from backend VirtualCamera (1280x720). */
export const CAM_W = 1280;
export const CAM_H = 720;

export interface BboxPercent {
  leftPct: number;
  topPct: number;
  widthPct: number;
  heightPct: number;
}

/**
 * Convert a backend bbox [x1, y1, x2, y2] in pixels to overlay percentages.
 * Returns null unless the bbox is exactly 4 finite numbers (Classical
 * detections carry no bbox and must keep the marker-only visualization).
 */
export function bboxToPercent(
  bbox: unknown,
  frameW: number = CAM_W,
  frameH: number = CAM_H
): BboxPercent | null {
  if (
    !Array.isArray(bbox) ||
    bbox.length !== 4 ||
    !bbox.every((v) => typeof v === 'number' && Number.isFinite(v)) ||
    !(frameW > 0) ||
    !(frameH > 0)
  ) {
    return null;
  }
  const [x1, y1, x2, y2] = bbox as [number, number, number, number];
  return {
    leftPct: (x1 / frameW) * 100,
    topPct: (y1 / frameH) * 100,
    widthPct: ((x2 - x1) / frameW) * 100,
    heightPct: ((y2 - y1) / frameH) * 100,
  };
}

export const fmt = (v: number | null | undefined, digits = 2, suffix = ''): string =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : `${v.toFixed(digits)}${suffix}`;

export const scenarioName = (exp: { scenario: string | { name?: string } }): string =>
  typeof exp.scenario === 'string' ? exp.scenario : exp.scenario?.name ?? 'custom';

/** Plain-text detector labels (DESIGN.md: ASCII brackets are the only
 * iconography — no emoji). */
export const detectorLabel = (detector: string): string =>
  detector === 'yolo' ? 'YOLO' : 'Classical';

export const detectorShortLabel = (detector: string): string =>
  detector === 'yolo' ? 'YOLO' : 'Classical';

export interface CompareRow {
  label: string;
  a: string;
  b: string;
}

/** Build the 12 side-by-side comparison rows verbatim from two detail payloads. */
export function buildCompareRows(a: ExperimentLike, b: ExperimentLike): CompareRow[] {
  const convergence = (exp: ExperimentLike): string =>
    exp.metrics.convergence_time_s === null
      ? '—'
      : `${fmt(exp.metrics.convergence_time_s)}s (frame ${exp.metrics.convergence_frame})`;
  return [
    { label: 'Detector', a: detectorLabel(a.detector), b: detectorLabel(b.detector) },
    { label: 'Scenario', a: scenarioName(a), b: scenarioName(b) },
    { label: 'Detection rate', a: fmt(a.metrics.detection_rate * 100, 1, '%'), b: fmt(b.metrics.detection_rate * 100, 1, '%') },
    { label: 'Avg confidence', a: fmt(a.metrics.avg_confidence * 100, 1, '%'), b: fmt(b.metrics.avg_confidence * 100, 1, '%') },
    { label: 'Initial error', a: `${fmt(a.metrics.initial_error)}px`, b: `${fmt(b.metrics.initial_error)}px` },
    { label: 'Final error', a: `${fmt(a.metrics.final_error)}px`, b: `${fmt(b.metrics.final_error)}px` },
    { label: 'Min error', a: `${fmt(a.metrics.minimum_error ?? a.metrics.min_error)}px`, b: `${fmt(b.metrics.minimum_error ?? b.metrics.min_error)}px` },
    { label: 'Convergence time', a: convergence(a), b: convergence(b) },
    { label: 'Convergence frame', a: fmt(a.metrics.convergence_frame, 0), b: fmt(b.metrics.convergence_frame, 0) },
    { label: 'LOCKED', a: a.metrics.locked_achieved ? 'Yes' : 'No', b: b.metrics.locked_achieved ? 'Yes' : 'No' },
    { label: 'Lock retention', a: `${fmt(a.metrics.lock_retention_s)}s`, b: `${fmt(b.metrics.lock_retention_s)}s` },
    { label: 'Lost count', a: `${a.metrics.lost_count}`, b: `${b.metrics.lost_count}` },
  ];
}

export interface GraphPoint {
  t: number;
  e: number;
}

export interface GraphSeries {
  id: string;
  label: string;
  color: string;
  /** Contiguous detected runs (data space); gaps where undetected. */
  segments: GraphPoint[][];
  /** Indices of isolated single detections (rendered as dots). */
  singles: number[];
  /** Index of first LOCKED record, or -1. */
  lockedIndex: number;
}

export interface GraphModel {
  xMax: number;
  yMax: number;
  series: GraphSeries[];
}

export const recordTime = (
  exp: { dt: number },
  rec: FrameRecord,
  i: number
): number => rec.time ?? (i + 1) * exp.dt;

/**
 * Build plottable series verbatim from raw records. Returns null when no
 * experiment carries per-frame records (caller shows the empty notice).
 */
export function buildGraphModel(
  experiments: GraphExperiment[],
  maxSeries: number = MAX_COMPARE_SERIES
): GraphModel | null {
  const shown = experiments.slice(0, maxSeries);
  const withData = shown.filter((exp) => exp.records && exp.records.length > 0);
  if (withData.length === 0) return null;

  let xMax = 0;
  let yMax = 0;
  for (const exp of withData) {
    const recs = exp.records as FrameRecord[];
    recs.forEach((rec, i) => {
      const t = recordTime(exp, rec, i);
      if (t > xMax) xMax = t;
      if (rec.detected && rec.error_px > yMax) yMax = rec.error_px;
    });
  }
  if (!(xMax > 0)) xMax = 1;
  if (!(yMax > 0)) yMax = 1;

  const series: GraphSeries[] = withData.map((exp, i) => {
    const recs = exp.records as FrameRecord[];
    const segments: GraphPoint[][] = [];
    let current: GraphPoint[] = [];
    recs.forEach((rec, j) => {
      if (!rec.detected) {
        if (current.length > 1) segments.push(current);
        current = [];
        return;
      }
      current.push({ t: recordTime(exp, rec, j), e: rec.error_px });
    });
    if (current.length > 1) segments.push(current);
    const singles = recs
      .map((rec, j) => ({ rec, j }))
      .filter(
        ({ rec, j }) =>
          rec.detected &&
          !(recs[j - 1]?.detected ?? false) &&
          !(recs[j + 1]?.detected ?? false)
      )
      .map(({ j }) => j);
    return {
      id: exp.experiment_id,
      label: `${detectorShortLabel(exp.detector)} · ${scenarioName(exp)}`,
      color: GRAPH_COLORS[i % GRAPH_COLORS.length],
      segments,
      singles,
      lockedIndex: recs.findIndex((rec) => rec.detected && rec.pat_state === 'LOCKED'),
    };
  });

  return { xMax, yMax, series };
}
