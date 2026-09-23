'use client';

import { useState } from 'react';
import styles from './MissionPanel.module.css';
import wb from './ResultWorkbench.module.css';
import { API_BASE } from '../config';
import { fmt, linkStateTone } from './experimentFormat';
import { ResultRows, WorkbenchNotice, WorkbenchSection } from './ResultWorkbench';

interface MissionPanelProps {
  detector: 'classical' | 'yolo';
}

interface MissionPhase {
  name: string;
  duration_s: number;
  end_state?: string;
  end_condition_met?: boolean;
}

/** Subset of the POST /mission/run report actually rendered here. */
interface MissionReport {
  mission: string;
  status: string;
  detector: string;
  seed: number;
  phase_order: string[];
  phases: MissionPhase[];
  final_pat_state: string;
  final_link_state: string;
  final_link_margin_db: number | null;
  final_pointing_error_px: number | null;
  recovery_time_s: number | null;
  search_to_reacquire_s: number | null;
}

const phaseDuration = (report: MissionReport, name: string): number | null => {
  const phase = report.phases.find((p) => p.name === name);
  return phase ? phase.duration_s : null;
};

export default function MissionPanel({ detector }: MissionPanelProps) {
  const [report, setReport] = useState<MissionReport | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/mission/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ detector, seed: 7 }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`POST /mission/run failed (${res.status}): ${text}`);
      }
      setReport((await res.json()) as MissionReport);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to run mission');
    } finally {
      setRunning(false);
    }
  };

  const statusTone = report
    ? report.status === 'COMPLETE'
      ? styles.statusComplete
      : styles.statusTimeout
    : '';
  const linkTone = report ? linkStateTone(report.final_link_state) : 'nodata';
  const linkToneClass =
    linkTone === 'nominal'
      ? styles.linkNominal
      : linkTone === 'marginal'
        ? styles.linkMarginal
        : linkTone === 'outage'
          ? styles.linkOutage
          : styles.linkNodata;

  return (
    <WorkbenchSection
      title="Mission Mode"
      meta={<span className={styles.missionDetector}>{detector}</span>}
    >
      <div className={styles.runRow}>
        <button
          className={styles.runButton}
          onClick={handleRun}
          disabled={running}
        >
          {running ? 'RUNNING MISSION…' : 'RUN MISSION'}
        </button>
      </div>

      {running && !report && (
        <WorkbenchNotice tone="loading">Running scripted mission…</WorkbenchNotice>
      )}
      {error && <WorkbenchNotice tone="error">{error}</WorkbenchNotice>}
      {!running && !error && !report && (
        <WorkbenchNotice tone="empty">
          <p>No mission run yet.</p>
          <span>
            Runs acquisition → lock → moving → disturbed → loss → search →
            reacquire → lock.
          </span>
        </WorkbenchNotice>
      )}

      {report && (
        <>
          <div className={styles.reportRow}>
            <span>Status</span>
            <span className={`${styles.statusBadge} ${statusTone}`}>
              {report.status}
            </span>
          </div>

          <div className={wb.title}>Phase timeline</div>
          <ResultRows
            rows={report.phase_order.map((name) => ({
              key: name,
              label: name,
              value: fmt(phaseDuration(report, name), 2, 's'),
            }))}
          />

          <ResultRows
            rows={[
              {
                key: 'recovery',
                label: 'Recovery time',
                value: fmt(report.recovery_time_s, 2, 's'),
              },
              {
                key: 'search-reacquire',
                label: 'Search → reacquire',
                value: fmt(report.search_to_reacquire_s, 3, 's'),
              },
              {
                key: 'final-pat',
                label: 'Final PAT state',
                value: report.final_pat_state,
              },
              {
                key: 'final-link',
                label: 'Final link',
                value: (
                  <span className={`${styles.linkBadge} ${linkToneClass}`}>
                    {report.final_link_state} (
                    {fmt(report.final_link_margin_db, 2, ' dB')})
                  </span>
                ),
              },
              {
                key: 'final-error',
                label: 'Final pointing error',
                value: fmt(report.final_pointing_error_px, 2, ' px'),
              },
            ]}
          />
        </>
      )}
    </WorkbenchSection>
  );
}
