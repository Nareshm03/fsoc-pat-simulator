'use client';

import { useState } from 'react';
import styles from './DatasetGenerator.module.css';

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

  const handleGenerate = async () => {
    if (!connected || isGenerating) return;

    setIsGenerating(true);
    setProgress(null);
    setStats(null);
    setError('');

    try {
      // Connect to dataset generation WebSocket
      const ws = new WebSocket('ws://localhost:8000/ws/dataset');

      ws.onopen = () => {
        console.log('[DATASET] WebSocket connected');
        // Send generate command
        ws.send(JSON.stringify({
          command: 'generate',
          num_images: numImages
        }));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('[DATASET]', data.type, data);

        if (data.type === 'ack') {
          console.log(`[DATASET] Generation started: ${data.num_images} images`);
        } else if (data.type === 'start') {
          setOutputDir(data.output_dir);
          console.log(`[DATASET] Output directory: ${data.output_dir}`);
        } else if (data.type === 'progress') {
          setProgress({
            current: data.current,
            total: data.total,
            percentage: data.percentage,
            split: data.split,
            filename: data.filename,
            detected: data.detected,
            confidence: data.confidence
          });
        } else if (data.type === 'complete') {
          setStats(data.stats);
          setOutputDir(data.output_dir);
          console.log('[DATASET] Generation complete', data.stats);
          ws.close();
          setIsGenerating(false);
        } else if (data.type === 'error') {
          setError(data.message);
          console.error('[DATASET] Error:', data.message);
          ws.close();
          setIsGenerating(false);
        }
      };

      ws.onerror = (error) => {
        console.error('[DATASET] WebSocket error:', error);
        setError('WebSocket connection error');
        setIsGenerating(false);
      };

      ws.onclose = () => {
        console.log('[DATASET] WebSocket closed');
        if (isGenerating && !stats) {
          setError('Connection closed unexpectedly');
          setIsGenerating(false);
        }
      };

    } catch (err) {
      console.error('[DATASET] Error:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
      setIsGenerating(false);
    }
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
            min="10"
            max="100000"
            step="10"
            value={numImages}
            onChange={(e) => setNumImages(parseInt(e.target.value))}
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
            <span>{progress.percentage.toFixed(1)}%</span>
          </div>
          
          <div className={styles.progressBar}>
            <div 
              className={styles.progressFill}
              style={{ width: `${progress.percentage}%` }}
            />
          </div>

          <div className={styles.progressDetails}>
            <div className={styles.progressItem}>
              <span className={styles.label}>Split:</span>
              <span className={`${styles.badge} ${styles[progress.split]}`}>
                {progress.split.toUpperCase()}
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
              <span>{(progress.confidence * 100).toFixed(1)}%</span>
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
                  ({(stats.train_count / stats.total_generated * 100).toFixed(0)}%)
                </span>
              </div>
            </div>
            
            <div className={styles.statItem}>
              <div className={styles.statLabel}>Validation Set</div>
              <div className={styles.statValue}>
                {stats.val_count}
                <span className={styles.percentage}>
                  ({(stats.val_count / stats.total_generated * 100).toFixed(0)}%)
                </span>
              </div>
            </div>
            
            <div className={styles.statItem}>
              <div className={styles.statLabel}>Test Set</div>
              <div className={styles.statValue}>
                {stats.test_count}
                <span className={styles.percentage}>
                  ({(stats.test_count / stats.total_generated * 100).toFixed(0)}%)
                </span>
              </div>
            </div>
            
            <div className={styles.statItem}>
              <div className={styles.statLabel}>Failed Detections</div>
              <div className={`${styles.statValue} ${stats.failed_detections > 0 ? styles.warning : styles.success}`}>
                {stats.failed_detections}
                <span className={styles.percentage}>
                  ({(stats.failed_detections / stats.total_generated * 100).toFixed(1)}%)
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
              <li>Train YOLO model: <code>yolo train data=dataset/dataset.yaml model=yolov8n.pt epochs=100</code></li>
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
