'use client';

import { useState, useEffect } from 'react';
import styles from './ExperimentHistory.module.css';

interface Experiment {
  experiment_id: number;
  start_time: string;
  duration: number;
  lock_retention: number;
  average_error: number;
  turbulence: number;
  vibration: number;
  camera_motion: number;
  noise: number;
  detector_type: string;
  status: string;
}

export default function ExperimentHistory() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  // TODO: Fetch from backend API when experiments are stored
  useEffect(() => {
    // No mock data - only show real experiments from backend
    // When backend experiment storage is implemented, fetch here:
    // fetch('/api/experiments').then(r => r.json()).then(setExperiments);
  }, []);

  const getPresetName = (exp: Experiment): string => {
    if (exp.turbulence === 0 && exp.vibration === 0 && exp.camera_motion === 0) {
      return 'Ideal';
    }
    if (exp.vibration > 1.5) return 'Vibration';
    if (exp.turbulence > 1.5) return 'Turbulence';
    if (exp.noise > 1.5) return 'High Noise';
    if (exp.turbulence > 1.5 && exp.vibration > 1.5) return 'Worst Case';
    if (exp.camera_motion > 1.5) return 'Camera Motion';
    return 'Custom';
  };

  const getLockClass = (retention: number): string => {
    if (retention >= 95) return styles.lockExcellent;
    if (retention >= 85) return styles.lockGood;
    if (retention >= 70) return styles.lockFair;
    return styles.lockPoor;
  };

  return (
    <div className={styles.historyContainer}>
      <div className={styles.historyHeader}>
        <h2>Experiment History</h2>
        <span className={styles.experimentCount}>
          {experiments.length} experiments
        </span>
      </div>

      <div className={styles.experimentList}>
        {experiments.map((exp) => (
          <div
            key={exp.experiment_id}
            className={`${styles.experimentCard} ${
              selectedId === exp.experiment_id ? styles.selected : ''
            }`}
            onClick={() => setSelectedId(exp.experiment_id)}
          >
            <div className={styles.expId}>
              #{exp.experiment_id.toString().padStart(3, '0')}
            </div>
            
            <div className={styles.expInfo}>
              <div className={styles.expName}>{getPresetName(exp)}</div>
              <div className={styles.expTime}>
                {new Date(exp.start_time).toLocaleTimeString()}
              </div>
            </div>

            <div className={styles.expMetrics}>
              <div className={styles.expMetric}>
                <span className={styles.metricLabel}>Lock:</span>
                <span className={`${styles.metricValue} ${getLockClass(exp.lock_retention)}`}>
                  {exp.lock_retention.toFixed(1)}%
                </span>
              </div>
              <div className={styles.expMetric}>
                <span className={styles.metricLabel}>Error:</span>
                <span className={styles.metricValue}>
                  {exp.average_error.toFixed(2)}px
                </span>
              </div>
            </div>

            <div className={styles.expDetector}>
              {exp.detector_type === 'yolo' ? '🤖 YOLO' : '👁 Classical'}
            </div>
          </div>
        ))}
      </div>

      {experiments.length === 0 && (
        <div className={styles.emptyState}>
          <p>No experiments yet</p>
          <p className={styles.emptyHint}>Run an experiment to see results here</p>
        </div>
      )}
    </div>
  );
}
