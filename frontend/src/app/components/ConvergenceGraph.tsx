'use client';

import styles from './ExperimentHistory.module.css';
import wb from './ResultWorkbench.module.css';
import {
  buildGraphModel,
  type FrameRecord,
  type GraphExperiment,
} from './experimentFormat';
import { WorkbenchNotice } from './ResultWorkbench';

export type { FrameRecord, GraphExperiment };

const W = 640;
const H = 360;
const MARGIN = { left: 56, right: 16, top: 16, bottom: 40 };

/**
 * Error-vs-time convergence graph (pure SVG, no chart dependency).
 * Plots raw per-frame records verbatim via buildGraphModel: x = record
 * time, y = error_px for detected frames (gaps where undetected).
 * No metric recalculation.
 */
export default function ConvergenceGraph({
  experiments,
}: {
  experiments: GraphExperiment[];
}) {
  const model = buildGraphModel(experiments);

  if (!model) {
    return (
      <div className={styles.graphPanel}>
        <div className={wb.title}>Convergence — error vs time</div>
        <WorkbenchNotice tone="empty">
          <p>No per-frame records available</p>
          <span>Experiment saved before records support.</span>
        </WorkbenchNotice>
      </div>
    );
  }

  const { xMax, yMax, series } = model;
  const iw = W - MARGIN.left - MARGIN.right;
  const ih = H - MARGIN.top - MARGIN.bottom;
  const x = (t: number): number => MARGIN.left + (t / xMax) * iw;
  const y = (e: number): number => MARGIN.top + ih - (e / yMax) * ih;

  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * xMax);
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax);
  // Adaptive precision: sub-pixel error ranges need more decimals.
  const yDigits = yMax < 1 ? 2 : yMax < 10 ? 1 : 0;

  return (
    <div className={styles.graphPanel}>
      <div className={wb.title}>Convergence — error vs time</div>
      <div className={styles.graphLegend}>
        {series.map((s) => (
          <span key={s.id} className={styles.legendItem}>
            <span
              className={styles.legendSwatch}
              style={{ background: s.color }}
            />
            {s.label}
          </span>
        ))}
      </div>
      <svg
        className={styles.graphSvg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Error versus time convergence graph"
      >
        {yTicks.map((v) => (
          <g key={`y${v}`}>
            <line
              x1={MARGIN.left}
              x2={W - MARGIN.right}
              y1={y(v)}
              y2={y(v)}
              strokeWidth={1}
              style={{ stroke: 'var(--hairline-strong)' }}
            />
            <text
              x={MARGIN.left - 6}
              y={y(v) + 4}
              textAnchor="end"
              fontSize={11}
              style={{ fill: 'var(--mute)' }}
            >
              {v.toFixed(yDigits)}
            </text>
          </g>
        ))}
        {xTicks.map((v) => (
          <text
            key={`x${v}`}
            x={x(v)}
            y={H - MARGIN.bottom + 18}
            textAnchor="middle"
            fontSize={11}
            style={{ fill: 'var(--mute)' }}
          >
            {v.toFixed(2)}
          </text>
        ))}
        <text
          x={MARGIN.left + iw / 2}
          y={H - 6}
          textAnchor="middle"
          fontSize={12}
          style={{ fill: 'var(--mute)' }}
        >
          time (s)
        </text>
        <text
          x={14}
          y={MARGIN.top + ih / 2}
          textAnchor="middle"
          fontSize={12}
          style={{ fill: 'var(--mute)' }}
          transform={`rotate(-90 14 ${MARGIN.top + ih / 2})`}
        >
          error_px
        </text>
        {series.map((s) => {
          const exp = experiments.find((e) => e.experiment_id === s.id);
          const recs = exp?.records ?? [];
          return (
            <g key={s.id}>
              {s.segments.map((pts, i) => (
                <polyline
                  key={i}
                  points={pts.map((p) => `${x(p.t).toFixed(2)},${y(p.e).toFixed(2)}`).join(' ')}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={2}
                />
              ))}
              {s.singles.map((j) => {
                const rec = recs[j];
                if (!rec) return null;
                const t = rec.time ?? (j + 1) * (exp?.dt ?? 1 / 30);
                return (
                  <circle
                    key={`s${j}`}
                    cx={x(t)}
                    cy={y(rec.error_px)}
                    r={2.5}
                    fill={s.color}
                  />
                );
              })}
              {s.lockedIndex >= 0 && recs[s.lockedIndex] && (() => {
                const rec = recs[s.lockedIndex];
                const t = rec.time ?? (s.lockedIndex + 1) * (exp?.dt ?? 1 / 30);
                return (
                  <g>
                    <circle
                      cx={x(t)}
                      cy={y(rec.error_px)}
                      r={5}
                      fill="none"
                      stroke={s.color}
                      strokeWidth={2}
                    />
                    <title>
                      {`LOCKED @ t=${t.toFixed(2)}s, err=${rec.error_px.toFixed(2)}px`}
                    </title>
                  </g>
                );
              })()}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
