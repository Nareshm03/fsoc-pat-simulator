import { describe, expect, it } from 'vitest';
import {
  applyControlAck,
  applyTelemetryLoopStatus,
  bboxToPercent,
  buildCompareRows,
  buildGraphModel,
  CAM_H,
  CAM_W,
  detectorLabel,
  detectorShortLabel,
  detectionRateTone,
  fmt,
  isPauseEnabled,
  isResumeEnabled,
  linkStateTone,
  patStateTone,
  recordTime,
  resolveYoloPipelinePhase,
  sanitizeNonJsonNumbers,
  scenarioName,
  YOLO_ACTIVE_FLASH_MS,
  YOLO_WARMUP_TIMEOUT_MS,
} from './experimentFormat';
import type { YoloPipelineInput } from './experimentFormat';
import type { ExperimentLike, FrameRecord } from './experimentFormat';

const rec = (
  partial: Partial<FrameRecord> & { error_px: number },
  i: number
): FrameRecord => ({
  detected: true,
  confidence: 0.9,
  pat_state: 'ACQUIRE',
  time: (i + 1) / 30,
  ...partial,
});

const metrics = {
  detection_rate: 1.0,
  avg_confidence: 0.93,
  initial_error: 250.63,
  final_error: 1.65,
  minimum_error: 0.49,
  min_error: 0.49,
  convergence_frame: 92,
  convergence_time_s: 92 / 30,
  locked_achieved: true,
  lock_retention_s: 4.93,
  lost_count: 0,
};

const expA: ExperimentLike = {
  experiment_id: 'aaa',
  detector: 'classical',
  scenario: 'stationary_ideal',
  dt: 1 / 30,
  metrics,
  records: [rec({ error_px: 10 }, 0), rec({ error_px: 2 }, 1)],
};

const expB: ExperimentLike = {
  experiment_id: 'bbb',
  detector: 'yolo',
  scenario: { name: 'moving_target' },
  dt: 1 / 30,
  metrics: {
    ...metrics,
    initial_error: null,
    final_error: null,
    minimum_error: null,
    min_error: null,
    convergence_frame: null,
    convergence_time_s: null,
    locked_achieved: false,
    lock_retention_s: 0,
  },
  records: null,
};

describe('fmt', () => {
  it('renders em-dash for null/undefined/NaN', () => {
    expect(fmt(null)).toBe('—');
    expect(fmt(undefined)).toBe('—');
    expect(fmt(NaN)).toBe('—');
  });
  it('rounds with digits and suffix', () => {
    expect(fmt(2.345)).toBe('2.35');
    expect(fmt(2.345, 1, '%')).toBe('2.3%');
    expect(fmt(92, 0)).toBe('92');
  });
});

describe('labels', () => {
  it('scenarioName handles string, object, and missing', () => {
    expect(scenarioName(expA)).toBe('stationary_ideal');
    expect(scenarioName(expB)).toBe('moving_target');
    expect(scenarioName({ scenario: {} })).toBe('custom');
  });
  it('detector labels', () => {
    expect(detectorLabel('yolo')).toBe('YOLO');
    expect(detectorLabel('classical')).toBe('Classical');
    expect(detectorShortLabel('yolo')).toBe('YOLO');
    expect(detectorShortLabel('other')).toBe('Classical');
  });
});

describe('buildCompareRows', () => {
  it('builds 12 verbatim rows with em-dashes for nulls', () => {
    const rows = buildCompareRows(expA, expB);
    expect(rows).toHaveLength(12);
    const byLabel = Object.fromEntries(rows.map((r) => [r.label, r]));
    expect(byLabel['Detector']).toEqual({ label: 'Detector', a: 'Classical', b: 'YOLO' });
    expect(byLabel['Detection rate'].a).toBe('100.0%');
    expect(byLabel['Initial error'].a).toBe('250.63px');
    expect(byLabel['Initial error'].b).toBe('—px');
    expect(byLabel['Min error'].a).toBe('0.49px');
    expect(byLabel['Convergence time'].a).toBe('3.07s (frame 92)');
    expect(byLabel['Convergence time'].b).toBe('—');
    expect(byLabel['LOCKED'].b).toBe('No');
    expect(byLabel['Lock retention'].a).toBe('4.93s');
    expect(byLabel['Lost count'].a).toBe('0');
  });
});

describe('recordTime', () => {
  it('prefers record time, falls back to (i+1)*dt', () => {
    expect(recordTime({ dt: 0.5 }, rec({ error_px: 1, time: 3.25 }, 0), 0)).toBe(3.25);
    expect(recordTime({ dt: 0.5 }, rec({ error_px: 1, time: null }, 2), 2)).toBe(1.5);
  });
});

describe('buildGraphModel', () => {
  it('returns null without records', () => {
    expect(buildGraphModel([expB])).toBeNull();
    expect(buildGraphModel([])).toBeNull();
  });
  it('maps bounds, segments, singles, and LOCKED index', () => {
    const records: FrameRecord[] = [
      rec({ error_px: 100, pat_state: 'ACQUIRE' }, 0),
      rec({ error_px: 50, pat_state: 'ACQUIRE' }, 1),
      { detected: false, confidence: 0, error_px: 0, pat_state: 'REACQUIRE', time: 3 / 30 },
      rec({ error_px: 20, pat_state: 'ACQUIRE' }, 3),
      rec({ error_px: 4, pat_state: 'LOCKED' }, 4),
      rec({ error_px: 3, pat_state: 'LOCKED' }, 5),
    ];
    const model = buildGraphModel([{ ...expA, records }]);
    expect(model).not.toBeNull();
    expect(model!.xMax).toBeCloseTo(6 / 30);
    expect(model!.yMax).toBe(100);
    // Gap at index 2 splits the line into two segments.
    expect(model!.series).toHaveLength(1);
    expect(model!.series[0].segments).toHaveLength(2);
    expect(model!.series[0].segments[0]).toHaveLength(2);
    expect(model!.series[0].segments[1]).toHaveLength(3);
    expect(model!.series[0].singles).toEqual([]);
    expect(model!.series[0].lockedIndex).toBe(4);
    expect(model!.series[0].label).toBe('Classical · stationary_ideal');
  });
  it('marks isolated detections as singles and caps at two series', () => {
    const lone: FrameRecord[] = [
      { detected: false, confidence: 0, error_px: 0, pat_state: 'SEARCH', time: 1 / 30 },
      rec({ error_px: 9, pat_state: 'ACQUIRE' }, 1),
      { detected: false, confidence: 0, error_px: 0, pat_state: 'SEARCH', time: 3 / 30 },
    ];
    const model = buildGraphModel([
      { ...expA, records: lone },
      { ...expB, records: lone, experiment_id: 'c2' },
      { ...expA, records: lone, experiment_id: 'c3' },
    ]);
    expect(model!.series).toHaveLength(2);
    expect(model!.series[0].segments).toEqual([]);
    expect(model!.series[0].singles).toEqual([1]);
    expect(model!.series[0].lockedIndex).toBe(-1);
  });
});

describe('bboxToPercent', () => {
  it('converts a full-frame box to 0/0/100/100', () => {
    expect(bboxToPercent([0, 0, CAM_W, CAM_H])).toEqual({
      leftPct: 0,
      topPct: 0,
      widthPct: 100,
      heightPct: 100,
    });
  });
  it('converts a centered 20x20 box around (640, 360)', () => {
    // Mirrors backend contract: bbox center == reported detection x/y.
    const box = bboxToPercent([630, 350, 650, 370]);
    expect(box).not.toBeNull();
    expect(box!.leftPct).toBeCloseTo((630 / CAM_W) * 100, 9);
    expect(box!.topPct).toBeCloseTo((350 / CAM_H) * 100, 9);
    expect(box!.widthPct).toBeCloseTo((20 / CAM_W) * 100, 9);
    expect(box!.heightPct).toBeCloseTo((20 / CAM_H) * 100, 9);
    const cxPct = box!.leftPct + box!.widthPct / 2;
    const cyPct = box!.topPct + box!.heightPct / 2;
    expect(cxPct).toBeCloseTo(50, 9);
    expect(cyPct).toBeCloseTo(50, 9);
  });
  it('rejects non-bbox values (Classical path stays marker-only)', () => {
    expect(bboxToPercent(null)).toBeNull();
    expect(bboxToPercent(undefined)).toBeNull();
    expect(bboxToPercent([1, 2, 3])).toBeNull();
    expect(bboxToPercent([1, 2, 3, 4, 5])).toBeNull();
    expect(bboxToPercent([0, 0, NaN, 10])).toBeNull();
    expect(bboxToPercent(['a', 0, 10, 10])).toBeNull();
    expect(bboxToPercent([0, 0, 10, 10], 0, CAM_H)).toBeNull();
  });
});

describe('linkStateTone', () => {
  it('maps known states and falls back to nodata', () => {
    expect(linkStateTone('NOMINAL')).toBe('nominal');
    expect(linkStateTone('MARGINAL')).toBe('marginal');
    expect(linkStateTone('OUTAGE')).toBe('outage');
    expect(linkStateTone(null)).toBe('nodata');
    expect(linkStateTone(undefined)).toBe('nodata');
    expect(linkStateTone('nominal')).toBe('nodata');
    expect(linkStateTone('')).toBe('nodata');
  });
});

describe('sanitizeNonJsonNumbers', () => {
  it('replaces bare non-JSON tokens with null', () => {
    expect(sanitizeNonJsonNumbers('{"a": Infinity}')).toBe('{"a": null}');
    expect(sanitizeNonJsonNumbers('{"a": -Infinity, "b": NaN}')).toBe(
      '{"a": null, "b": null}'
    );
    expect(JSON.parse(sanitizeNonJsonNumbers('{"a": Infinity}'))).toEqual({
      a: null,
    });
  });
  it('leaves valid JSON untouched', () => {
    const valid = '{"a": 1.5, "b": "Infinity", "c": null, "d": [1, -2.5]}';
    expect(sanitizeNonJsonNumbers(valid)).toBe(valid);
  });
  it('preserves tokens inside strings (e.g. base64 image data)', () => {
    // 'SW5maW5pdHk=' is base64 for the text "Infinity".
    const withImage =
      '{"image": "SW5maW5pdHk=", "pointing_loss_db": Infinity}';
    expect(sanitizeNonJsonNumbers(withImage)).toBe(
      '{"image": "SW5maW5pdHk=", "pointing_loss_db": null}'
    );
  });
  it('does not touch identifier-like substrings', () => {
    expect(sanitizeNonJsonNumbers('{"xInfinity": 1}')).toBe('{"xInfinity": 1}');
  });
});

describe('resolveYoloPipelinePhase', () => {
  const live = {
    selectedDetector: 'yolo',
    activeDetector: 'yolo',
    loopLive: true,
    seenYoloDetection: false,
    warmupExpired: false,
  } satisfies YoloPipelineInput;

  it('reports warming while YOLO is selected, confirmed, live, unseen, unexpired', () => {
    expect(resolveYoloPipelinePhase(live)).toBe('warming');
  });
  it('reports settled once a nominal YOLO detection has been seen', () => {
    expect(
      resolveYoloPipelinePhase({ ...live, seenYoloDetection: true })
    ).toBe('settled');
  });
  it('reports settled after the warmup timeout even with no detections (genuine loss reads normally)', () => {
    expect(resolveYoloPipelinePhase({ ...live, warmupExpired: true })).toBe(
      'settled'
    );
  });
  it('reports inactive for classical selection', () => {
    expect(
      resolveYoloPipelinePhase({ ...live, selectedDetector: 'classical' })
    ).toBe('inactive');
  });
  it('reports inactive while the backend still runs classical (switch pending)', () => {
    expect(
      resolveYoloPipelinePhase({ ...live, activeDetector: 'classical' })
    ).toBe('inactive');
  });
  it('reports inactive when the loop is idle or paused', () => {
    expect(resolveYoloPipelinePhase({ ...live, loopLive: false })).toBe(
      'inactive'
    );
  });
  it('keeps timeouts ordered: flash < warmup, warmup >> pipeline fill', () => {
    // Pipeline fill is ~0.1-0.3 s of observations; cold model load is
    // seconds (first-tick cost measured in backend perf tests).
    expect(YOLO_ACTIVE_FLASH_MS).toBeLessThan(YOLO_WARMUP_TIMEOUT_MS);
    expect(YOLO_WARMUP_TIMEOUT_MS).toBeGreaterThan(1000);
  });
});

describe('detectionRateTone', () => {
  it('grades the lock-quality scale at 95/85/70 boundaries', () => {
    expect(detectionRateTone(100)).toBe('excellent');
    expect(detectionRateTone(95)).toBe('excellent');
    expect(detectionRateTone(94.9)).toBe('good');
    expect(detectionRateTone(85)).toBe('good');
    expect(detectionRateTone(84.9)).toBe('fair');
    expect(detectionRateTone(70)).toBe('fair');
    expect(detectionRateTone(69.9)).toBe('poor');
    expect(detectionRateTone(0)).toBe('poor');
  });
  it('treats non-finite input as poor (never throws on wire data)', () => {
    expect(detectionRateTone(NaN)).toBe('poor');
    expect(detectionRateTone(Infinity)).toBe('poor');
  });
});

describe('patStateTone', () => {
  it('maps precision states to the success tone', () => {
    expect(patStateTone('LOCKED')).toBe('locked');
    expect(patStateTone('FINE_TRACK')).toBe('fine_track');
  });
  it('maps transitional states to warning/accent tones', () => {
    expect(patStateTone('ACQUIRE')).toBe('acquire');
    expect(patStateTone('SEARCH')).toBe('search');
    expect(patStateTone('REACQUIRE')).toBe('reacquire');
  });
  it('is case-insensitive and falls back to idle', () => {
    expect(patStateTone('locked')).toBe('locked');
    expect(patStateTone(null)).toBe('idle');
    expect(patStateTone(undefined)).toBe('idle');
    expect(patStateTone('SOMETHING_ELSE')).toBe('idle');
  });
});

describe('pause/resume control-plane state flow', () => {
  it('successful pause ack pauses a running loop (PAUSE off, RESUME on)', () => {
    const next = applyControlAck({ running: true, paused: false }, 'pause');
    expect(next).toEqual({ running: true, paused: true });
    expect(isPauseEnabled(next)).toBe(false);
    expect(isResumeEnabled(next)).toBe(true);
  });
  it('successful resume ack clears paused without touching running', () => {
    const next = applyControlAck({ running: true, paused: true }, 'resume');
    expect(next).toEqual({ running: true, paused: false });
    expect(isPauseEnabled(next)).toBe(true);
    expect(isResumeEnabled(next)).toBe(false);
  });
  it('failed pause ack leaves an idle loop untouched (same reference)', () => {
    const prev = { running: false, paused: false };
    expect(applyControlAck(prev, 'pause')).toBe(prev);
  });
  it('failed resume ack leaves an idle loop untouched (same reference)', () => {
    const prev = { running: false, paused: false };
    expect(applyControlAck(prev, 'resume')).toBe(prev);
  });
  it('telemetry resynchronizes optimistic state when ticks resume', () => {
    const optimistic = applyControlAck(
      { running: true, paused: false },
      'pause'
    );
    expect(optimistic.paused).toBe(true);
    const resynced = applyTelemetryLoopStatus(true, false);
    expect(resynced).toEqual({ running: true, paused: false });
    expect(isPauseEnabled(resynced)).toBe(true);
  });
  it('reconnect never turns a genuinely paused loop into running', () => {
    applyControlAck({ running: true, paused: false }, 'pause');
    // Drop carries no state change; first post-reconnect telemetry wins.
    const stillPaused = applyTelemetryLoopStatus(true, true);
    expect(stillPaused).toEqual({ running: true, paused: true });
    expect(isResumeEnabled(stillPaused)).toBe(true);
  });
  it('start/stop/reset flows keep sole ownership of the running flag', () => {
    // Buttons derive from the same status: idle disables both pause and
    // resume, so start/stop/reset behavior is unchanged by pause optimism.
    expect(isPauseEnabled({ running: false, paused: false })).toBe(false);
    expect(isResumeEnabled({ running: false, paused: false })).toBe(false);
    expect(isPauseEnabled({ running: true, paused: false })).toBe(true);
  });
});
