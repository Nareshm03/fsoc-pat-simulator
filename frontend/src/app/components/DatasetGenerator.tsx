'use client';

import { useEffect, useRef, useState } from 'react';
import styles from './DatasetGenerator.module.css';
import { WS_BASE } from '../config';

const MIN_IMAGES = 10;
const MAX_IMAGES = 100000;

const clampImages = (value: number): number => {
  if (!Number.isFinite(value)) return 300;
  return Math.min(MAX_IMAGES, Math.max(MIN_IMAGES, Math.trunc(value)));
};

/** Zero-safe percentage string (avoids NaN% when totals are 0). */
const pct = (part: number, total: number, digits: number): string => {
  if (!Number.isFinite(part) || !Number.isFinite(total) || total <= 0) return `0.${'0'.repeat(digits)}%`;
  return `${((part / total) * 100).toFixed(digits)}%`;
};

interface DatasetGeneratorProps {
  connected: boolean;
}

interface GenerationProgress {
  current: number;
  total: number;
  percentage: number;
  split: string;
  filename: string;
  detected: boolean;
  confidence: number;
}

interface GenerationStats {
  total_generated: number;
  train_count: number;
  val_count: number;
  test_count: number;
  failed_detections: number;
}

export default function DatasetGenerator({ connected }: DatasetGeneratorProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [numImages, setNumImages] = useState(300);  // Pilot dataset
  const [progress, setProgress] = useState<GenerationProgress | null>(null);
  const [stats, setStats] = useState<GenerationStats | null>(null);
  const [outputDir, setOutputDir] = useState<string>('');
  const [error, setError] = useState<string>('');
  const wsRef = useRef<WebSocket | null>(null);
  const finishedRef = useRef(false);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  const finish = (message: string | null) => {
    finishedRef.current = true;
    if (message && mountedRef.current) setError(message);
    try {
      wsRef.current?.close();
    } catch {
      /* already closed */
    }
    wsRef.current = null;
    if (mountedRef.current) setIsGenerating(false);
  };

  const handleCancel = () => {
    finish('Generation cancelled.');
  };

  const handleGenerate = async () => {
    if (!connected || isGenerating) return;

    const count = clampImages(numImages);
    setNumImages(count);
    setIsGenerating(true);
    setProgress(null);
    setStats(null);
    setError('');
    finishedRef.current = false;

    let ws: WebSocket;
    try {
      // Connect to dataset generation WebSocket
      ws = new WebSocket(`${WS_BASE}/ws/dataset`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setIsGenerating(false);
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => {
      // Send generate command
      try {
        ws.send(JSON.stringify({
          command: 'generate',
          num_images: count
        }));
      } catch (err) {
        finish(err instanceof Error ? err.message : 'Failed to start generation');
      }
    };

    ws.onmessage = (event) => {
      let data: {
        type?: string;
        num_images?: number;
        output_dir?: string;
        current?: number;
        total?: number;
        percentage?: number;
        split?: string;
        filename?: string;
        detected?: boolean;
        confidence?: number;
        stats?: GenerationStats;
        message?: string;
      };
      try {
        data = JSON.parse(event.data);
      } catch {
        return; // ignore malformed frames
      }
      if (!data || typeof data.type !== 'string') return;

      if (data.type === 'start') {
        setOutputDir(data.output_dir ?? '');
      } else if (data.type === 'progress') {
        setProgress({
          current: data.current ?? 0,
          total: data.total ?? count,
          percentage: data.percentage ?? 0,
          split: data.split ?? 'train',
          filename: data.filename ?? '',
          detected: !!data.detected,
          confidence: data.confidence ?? 0
        });
      } else if (data.type === 'complete') {
        if (data.stats) setStats(data.stats);
        setOutputDir(data.output_dir ?? '');
        finish(null);
      } else if (data.type === 'error') {
        finish(data.message ?? 'Dataset generation failed');
      }
      // 'ack' and unknown types need no UI action.
    };

    ws.onerror = () => {
      finish('WebSocket connection error');
    };

    ws.onclose = () => {
      wsRef.current = null;
      // finishedRef distinguishes clean completion/cancel from a drop.
      if (!finishedRef.current) {
        finish('Connection closed unexpectedly');
      }
    };
  };

  return (
    <div className={styles.container}>
      <h2>Dataset Generator</h2>
      
      <div className={styles.controls}>
        <div className={styles.inputGroup}>
          <label htmlFor="numImages">Number of Images:</label>
          <input
            id="numImages"
            type="number"
            min={MIN_IMAGES}
            max={MAX_IMAGES}
            step="10"
            value={numImages}
            onChange={(e) => {
              const parsed = parseInt(e.target.value, 10);
              setNumImages(Number.isFinite(parsed) ? parsed : MIN_IMAGES);
            }}
            onBlur={() => setNumImages((n) => clampImages(n))}
            disabled={isGenerating}
            className={styles.numberInput}
          />
        </div>

        <button
          className={styles.generateBtn}
          onClick={handleGenerate}
          disabled={!connected || isGenerating}
          title={!connected ? "Connect to backend first" : "Generate training dataset"}
        >
          {isGenerating ? 'GENERATING...' : 'GENERATE DATASET'}
        </button>
        {isGenerating && (
          <button
            className={styles.generateBtn}
            onClick={handleCancel}
            title="Cancel dataset generation"
          >
            CANCEL
          </button>
        )}
      </div>

      {error && (
        <div className={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {progress && (
        <div className={styles.progress}>
          <div className={styles.progressHeader}>
            <span>Generating: {progress.current} / {progress.total}</span>
            <span>{(Number.isFinite(progress.percentage) ? progress.percentage : 0).toFixed(1)}%</span>
          </div>

          <div className={styles.progressBar}>
            <div
              className={styles.progressFill}
              style={{ width: `${Number.isFinite(progress.percentage) ? progress.percentage : 0}%` }}
            />
          </div>

          <div className={styles.progressDetails}>
            <div className={styles.progressItem}>
              <span className={styles.label}>Split:</span>
              <span className={`${styles.badge} ${styles[progress.split] ?? ''}`}>
                {(progress.split || 'train').toUpperCase()}
              </span>
            </div>
            <div className={styles.progressItem}>
              <span className={styles.label}>File:</span>
              <span className={styles.filename}>{progress.filename}</span>
            </div>
            <div className={styles.progressItem}>
              <span className={styles.label}>Detected:</span>
              <span className={progress.detected ? styles.success : styles.warning}>
                {progress.detected ? 'YES' : 'NO'}
              </span>
            </div>
            <div className={styles.progressItem}>
              <span className={styles.label}>Confidence:</span>
              <span>{((Number.isFinite(progress.confidence) ? progress.confidence : 0) * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}

      {stats && (
        <div className={styles.stats}>
          <h3>Generation Complete! ✓</h3>

          <div className={styles.statsGrid}>
            <div className={styles.statItem}>
              <div className={styles.statLabel}>Total Generated</div>
              <div className={styles.statValue}>{stats.total_generated}</div>
            </div>

            <div className={styles.statItem}>
              <div className={styles.statLabel}>Train Set</div>
              <div className={styles.statValue}>
                {stats.train_count}
                <span className={styles.percentage}>
                  ({pct(stats.train_count, stats.total_generated, 0)})
                </span>
              </div>
            </div>

            <div className={styles.statItem}>
              <div className={styles.statLabel}>Validation Set</div>
              <div className={styles.statValue}>
                {stats.val_count}
                <span className={styles.percentage}>
                  ({pct(stats.val_count, stats.total_generated, 0)})
                </span>
              </div>
            </div>

            <div className={styles.statItem}>
              <div className={styles.statLabel}>Test Set</div>
              <div className={styles.statValue}>
                {stats.test_count}
                <span className={styles.percentage}>
                  ({pct(stats.test_count, stats.total_generated, 0)})
                </span>
              </div>
            </div>

            <div className={styles.statItem}>
              <div className={styles.statLabel}>Failed Detections</div>
              <div className={`${styles.statValue} ${stats.failed_detections > 0 ? styles.warning : styles.success}`}>
                {stats.failed_detections}
                <span className={styles.percentage}>
                  ({pct(stats.failed_detections, stats.total_generated, 1)})
                </span>
              </div>
            </div>
          </div>

          {outputDir && (
            <div className={styles.outputInfo}>
              <strong>Output Directory:</strong>
              <code className={styles.path}>{outputDir}</code>
            </div>
          )}

          <div className={styles.nextSteps}>
            <h4>Next Steps:</h4>
            <ol>
              <li>Check the <code>dataset/</code> folder in your backend directory</li>
              <li>Review the generated <code>dataset.yaml</code> file</li>
              <li>Train YOLO model: <code>yolo train data=dataset/dataset.yaml model=yolo11n.pt epochs=100</code></li>
            </ol>
          </div>
        </div>
      )}

      <div className={styles.info}>
        <h4>Dataset Information</h4>
        <ul>
          <li><strong>Automatic Split:</strong> 80% Train / 10% Val / 10% Test</li>
          <li><strong>Randomized:</strong> Position, size, brightness, turbulence, noise, vibration, blur</li>
          <li><strong>Format:</strong> PNG images + YOLO TXT labels</li>
          <li><strong>No Manual Labeling:</strong> Fully automated with ground truth</li>
        </ul>
      </div>
    </div>
  );
}
