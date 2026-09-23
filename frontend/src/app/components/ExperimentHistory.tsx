'use client';

import { useState, useEffect, useCallback } from 'react';
import styles from './ExperimentHistory.module.css';
import wb from './ResultWorkbench.module.css';
import ConvergenceGraph from './ConvergenceGraph';
import { API_BASE } from '../config';
import {
  buildCompareRows,
  detectionRateTone,
  detectorLabel,
  fmt,
  scenarioName,
  type FrameRecord,
} from './experimentFormat';
import {
  ResultRows,
  WorkbenchNotice,
  WorkbenchSection,
} from './ResultWorkbench';

interface ExperimentMetrics {
  frames: number;
  detections: number;
  misses: number;
  detection_rate: number;
  avg_confidence: number;
  min_confidence: number;
  max_confidence: number;
  initial_error: number | null;
  final_error: number | null;
  minimum_error: number | null;
  min_error: number | null;
  convergence_frame: number | null;
  convergence_time_s: number | null;
  locked_achieved: boolean;
  lock_time_s: number | null;
  lock_retention_s: number;
  lost_count: number;
}

interface Experiment {
  experiment_id: string;
  timestamp: string;
  detector: string;
  scenario: string | { name?: string };
  dt: number;
  duration_s: number;
  duration: number;
  config: {
    ticks?: number;
    seed?: number;
    [key: string]: unknown;
  };
  metrics: ExperimentMetrics;
  records?: FrameRecord[] | null;
}

interface ScenarioPreset {
  label: string;
  value: Record<string, number | string>;
}

const SCENARIO_PRESETS: ScenarioPreset[] = [
  {
    label: 'Stationary + Ideal',
    value: {
      name: 'stationary_ideal', kind: 'stationary',
      az_deg: 5.0, el_deg: -8.0,
      turbulence: 0.0, vibration: 0.0, camera_motion: 0.0, sensor_noise: 0.0,
    },
  },
  {
    label: 'Stationary + Turbulence',
    value: {
      name: 'stationary_turbulence', kind: 'stationary',
      az_deg: 5.0, el_deg: -8.0,
      turbulence: 2.0, vibration: 0.0, camera_motion: 0.0, sensor_noise: 0.0,
    },
  },
  {
    label: 'Stationary + Vibration',
    value: {
      name: 'stationary_vibration', kind: 'stationary',
      az_deg: 5.0, el_deg: -8.0,
      turbulence: 0.0, vibration: 2.0, camera_motion: 0.0, sensor_noise: 0.0,
    },
  },
  {
    label: 'Stationary + Camera Motion',
    value: {
      name: 'stationary_camera_motion', kind: 'stationary',
      az_deg: 5.0, el_deg: -8.0,
      turbulence: 0.0, vibration: 0.0, camera_motion: 2.0, sensor_noise: 0.0,
    },
  },
  {
    label: 'Stationary + High Noise',
    value: {
      name: 'stationary_high_noise', kind: 'stationary',
      az_deg: 5.0, el_deg: -8.0,
      turbulence: 0.0, vibration: 0.0, camera_motion: 0.0, sensor_noise: 2.0,
    },
  },
  {
    label: 'Moving Target',
    value: {
      name: 'moving_target', kind: 'moving',
      turbulence: 0.5, vibration: 0.5, camera_motion: 0.5, sensor_noise: 0.5,
    },
  },
];

function CompareTable({ a, b }: { a: Experiment; b: Experiment }) {
  const shortId = (exp: Experiment): string => `#${exp.experiment_id.slice(0, 8)}`;
  // Rows come verbatim from the GET detail payloads (see experimentFormat).
  const rows = buildCompareRows(a, b);
  return (
    <div className={styles.comparePanel}>
      <div className={wb.title}>
        Comparison — {shortId(a)} vs {shortId(b)}
      </div>
      <table className={styles.compareTable}>
        <thead>
          <tr>
            <th>Metric</th>
            <th>{shortId(a)}</th>
            <th>{shortId(b)}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <td>{row.label}</td>
              <td>{row.a}</td>
              <td>{row.b}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ExperimentHistory({
  yoloAvailable = true,
}: {
  yoloAvailable?: boolean;
}) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selected, setSelected] = useState<Experiment | null>(null);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [runDetector, setRunDetector] = useState<'classical' | 'yolo'>('classical');
  const [runScenario, setRunScenario] = useState(0);
  const [runSeed, setRunSeed] = useState(0);
  const [runTicks, setRunTicks] = useState(150);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [compareData, setCompareData] = useState<Experiment[]>([]);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [compareHint, setCompareHint] = useState<string | null>(null);

  const fetchList = useCallback(async () => {
    setListLoading(true);
    setListError(null);
    try {
      const res = await fetch(`${API_BASE}/experiments`);
      if (!res.ok) throw new Error(`GET /experiments failed (${res.status})`);
      setExperiments(await res.json());
    } catch (e) {
      setListError(e instanceof Error ? e.message : 'Failed to load experiments');
    } finally {
      setListLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchList();
  }, [fetchList]);

  const handleSelect = useCallback(async (id: string) => {
    setSelectedId(id);
    setSelected(null);
    setDetailLoading(true);
    setDetailError(null);
    try {
      const res = await fetch(`${API_BASE}/experiments/${encodeURIComponent(id)}`);
      if (!res.ok) throw new Error(`GET /experiments/${id} failed (${res.status})`);
      setSelected(await res.json());
    } catch (e) {
      setDetailError(e instanceof Error ? e.message : 'Failed to load experiment');
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const handleRun = async () => {
    // Validate locally so the backend only ever sees sane values.
    const ticks = Number.isFinite(runTicks)
      ? Math.min(2000, Math.max(1, Math.trunc(runTicks)))
      : 150;
    const seed = Number.isFinite(runSeed) ? Math.trunc(runSeed) : 0;
    const preset = SCENARIO_PRESETS[runScenario] ?? SCENARIO_PRESETS[0];
    setRunTicks(ticks);
    setRunSeed(seed);
    setRunning(true);
    setRunError(null);
    try {
      const res = await fetch(`${API_BASE}/experiments/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          detector: runDetector,
          scenario: preset.value,
          seed,
          ticks,
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`POST /experiments/run failed (${res.status}): ${text}`);
      }
      const saved: Experiment = await res.json();
      await fetchList();
      // Show the exact saved result via a fresh GET (proves the round-trip).
      await handleSelect(saved.experiment_id);
    } catch (e) {
      setRunError(e instanceof Error ? e.message : 'Failed to run experiment');
    } finally {
      setRunning(false);
    }
  };

  const toggleCompare = (id: string) => {
    if (compareIds.includes(id)) {
      setCompareHint(null);
      setCompareIds(compareIds.filter((x) => x !== id));
    } else if (compareIds.length >= 2) {
      setCompareHint('Comparison is limited to 2 — deselect one first.');
    } else {
      setCompareHint(null);
      setCompareIds([...compareIds, id]);
    }
    setCompareData([]);
    setCompareError(null);
  };

  const handleCompare = async () => {
    if (compareIds.length !== 2) return;
    setComparing(true);
    setCompareError(null);
    setCompareData([]);
    try {
      // Detail payloads only — displayed verbatim, never recalculated.
      const results = await Promise.all(
        compareIds.map(async (id) => {
          const res = await fetch(`${API_BASE}/experiments/${encodeURIComponent(id)}`);
          if (!res.ok) throw new Error(`GET /experiments/${id} failed (${res.status})`);
          return (await res.json()) as Experiment;
        })
      );
      setCompareData(results);
    } catch (e) {
      setCompareError(e instanceof Error ? e.message : 'Failed to load comparison');
    } finally {
      setComparing(false);
    }
  };

  const clearCompare = () => {
    setCompareIds([]);
    setCompareData([]);
    setCompareError(null);
    setCompareHint(null);
  };

  const QUALITY_CLASS = {
    excellent: styles.lockExcellent,
    good: styles.lockGood,
    fair: styles.lockFair,
    poor: styles.lockPoor,
  } as const;
  const getQualityClass = (detectionRatePct: number): string =>
    QUALITY_CLASS[detectionRateTone(detectionRatePct)];

  return (
    <WorkbenchSection
      title="Experiment History"
      meta={
        <span className={styles.experimentCount}>
          {experiments.length} experiments
        </span>
      }
    >

      <div className={styles.runForm}>
        <div className={styles.formRow}>
          <label className={styles.formField}>
            Detector
            <select
              value={runDetector}
              onChange={(e) => setRunDetector(e.target.value as 'classical' | 'yolo')}
              disabled={running}
            >
              <option value="classical">Classical CV</option>
              <option value="yolo" disabled={!yoloAvailable}>
                YOLO11{yoloAvailable ? '' : ' (unavailable)'}
              </option>
            </select>
          </label>
          <label className={styles.formField}>
            Scenario
            <select
              value={runScenario}
              onChange={(e) => setRunScenario(Number(e.target.value))}
              disabled={running}
            >
              {SCENARIO_PRESETS.map((preset, i) => (
                <option key={preset.label} value={i}>
                  {preset.label}
                </option>
              ))}
            </select>
          </label>
          <label className={styles.formField}>
            Seed
            <input
              type="number"
              value={runSeed}
              onChange={(e) => setRunSeed(Number(e.target.value))}
              disabled={running}
            />
          </label>
          <label className={styles.formField}>
            Ticks
            <input
              type="number"
              min={1}
              max={2000}
              value={runTicks}
              onChange={(e) => setRunTicks(Number(e.target.value))}
              disabled={running}
            />
          </label>
          <button
            className={styles.runButton}
            onClick={handleRun}
            disabled={running}
          >
            {running ? 'RUNNING…' : 'RUN EXPERIMENT'}
          </button>
        </div>
        {runError && <WorkbenchNotice tone="error">{runError}</WorkbenchNotice>}
      </div>

      {listLoading && (
        <WorkbenchNotice tone="loading">Loading experiments…</WorkbenchNotice>
      )}
      {listError && <WorkbenchNotice tone="error">{listError}</WorkbenchNotice>}

      <div className={styles.experimentList}>
        {experiments.map((exp) => {
          const ratePct = exp.metrics.detection_rate * 100;
          return (
            <div
              key={exp.experiment_id}
              className={`${styles.experimentCard} ${
                selectedId === exp.experiment_id ? styles.selected : ''
              }`}
              onClick={() => handleSelect(exp.experiment_id)}
            >
              <div className={styles.expId}>
                #{exp.experiment_id.slice(0, 8)}
              </div>

              <div className={styles.expInfo}>
                <div className={styles.expName}>{scenarioName(exp)}</div>
                <div className={styles.expTime}>
                  {new Date(exp.timestamp).toLocaleString()}
                </div>
              </div>

              <div className={styles.expMetrics}>
                <div className={styles.expMetric}>
                  <span className={styles.metricLabel}>Detect:</span>
                  <span className={`${styles.metricValue} ${getQualityClass(ratePct)}`}>
                    {ratePct.toFixed(1)}%
                  </span>
                </div>
                <div className={styles.expMetric}>
                  <span className={styles.metricLabel}>Final err:</span>
                  <span className={styles.metricValue}>
                    {fmt(exp.metrics.final_error)}px
                  </span>
                </div>
              </div>

              <div className={styles.expDetector}>
                {detectorLabel(exp.detector)}
              </div>

              <label
                className={styles.compareCheck}
                title={
                  compareIds.includes(exp.experiment_id)
                    ? 'Remove from comparison'
                    : compareIds.length >= 2
                      ? 'Comparison is limited to 2 experiments'
                      : 'Select for comparison'
                }
                onClick={(e) => e.stopPropagation()}
              >
                <input
                  type="checkbox"
                  checked={compareIds.includes(exp.experiment_id)}
                  disabled={!compareIds.includes(exp.experiment_id) && compareIds.length >= 2}
                  onChange={() => toggleCompare(exp.experiment_id)}
                />
                <span>Compare</span>
              </label>
            </div>
          );
        })}
      </div>

      <div className={styles.compareBar}>
        <span className={styles.compareCount}>
          {compareIds.length}/2 selected for comparison
        </span>
        <button
          className={styles.compareButton}
          onClick={handleCompare}
          disabled={compareIds.length !== 2 || comparing}
        >
          {comparing ? 'COMPARING…' : 'COMPARE'}
        </button>
        {compareIds.length > 0 && (
          <button className={styles.compareClear} onClick={clearCompare}>
            Clear
          </button>
        )}
      </div>
      {compareError && (
        <WorkbenchNotice tone="error">{compareError}</WorkbenchNotice>
      )}
      {compareHint && !compareError && (
        <WorkbenchNotice tone="loading">{compareHint}</WorkbenchNotice>
      )}

      {compareData.length === 2 && !comparing && (
        <CompareTable a={compareData[0]} b={compareData[1]} />
      )}

      {compareData.length === 2 && !comparing ? (
        <ConvergenceGraph experiments={compareData} />
      ) : (
        selected &&
        !detailLoading && <ConvergenceGraph experiments={[selected]} />
      )}

      {!listLoading && experiments.length === 0 && !listError && (
        <WorkbenchNotice tone="empty" align="center">
          <p>No experiments yet</p>
          <span>Run an experiment to see results here</span>
        </WorkbenchNotice>
      )}

      {detailLoading && (
        <WorkbenchNotice tone="loading">Loading experiment…</WorkbenchNotice>
      )}
      {detailError && (
        <WorkbenchNotice tone="error">{detailError}</WorkbenchNotice>
      )}

      {selected && !detailLoading && (
        <>
          <div className={wb.title}>
            #{selected.experiment_id.slice(0, 8)} — {scenarioName(selected)}
          </div>
          <ResultRows
            rows={[
              { key: 'detector', label: 'Detector', value: detectorLabel(selected.detector) },
              { key: 'scenario', label: 'Scenario', value: scenarioName(selected) },
              {
                key: 'duration',
                label: 'Duration',
                value: `${fmt(selected.duration_s)}s (${selected.metrics.frames} frames)`,
              },
              {
                key: 'locked',
                label: 'LOCKED',
                value: selected.metrics.locked_achieved
                  ? `Yes (${fmt(selected.metrics.lock_retention_s)}s)`
                  : 'No',
              },
              {
                key: 'detection-rate',
                label: 'Detection rate',
                value: fmt(selected.metrics.detection_rate * 100, 1, '%'),
              },
              {
                key: 'convergence',
                label: 'Convergence time',
                value:
                  selected.metrics.convergence_time_s === null
                    ? '—'
                    : `${fmt(selected.metrics.convergence_time_s)}s (frame ${selected.metrics.convergence_frame})`,
              },
              {
                key: 'final-error',
                label: 'Final error',
                value: `${fmt(selected.metrics.final_error)}px`,
              },
              {
                key: 'min-error',
                label: 'Min error',
                value: `${fmt(selected.metrics.minimum_error ?? selected.metrics.min_error)}px`,
              },
              {
                key: 'confidence',
                label: 'Confidence',
                value: fmt(selected.metrics.avg_confidence * 100, 1, '%'),
              },
            ]}
          />
        </>
      )}
    </WorkbenchSection>
  );
}
