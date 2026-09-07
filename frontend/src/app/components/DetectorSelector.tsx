'use client';

import styles from './DetectorSelector.module.css';

interface DetectorSelectorProps {
  selectedDetector: 'classical' | 'yolo';
  onDetectorChange: (detector: 'classical' | 'yolo') => void;
  yoloAvailable?: boolean;
}

export default function DetectorSelector({
  selectedDetector,
  onDetectorChange,
  yoloAvailable = false
}: DetectorSelectorProps) {
  return (
    <div className={styles.selectorContainer}>
      <div className={styles.selectorLabel}>DETECTOR</div>
      <div className={styles.options}>
        <label className={styles.radioOption}>
          <input
            type="radio"
            name="detector"
            value="classical"
            checked={selectedDetector === 'classical'}
            onChange={() => onDetectorChange('classical')}
          />
          <span className={styles.radioLabel}>
            <span className={styles.radioIcon}>👁</span>
            Classical CV
          </span>
          <span className={styles.radioDescription}>
            OpenCV-based detection (threshold + contours)
          </span>
        </label>

        <label 
          className={`${styles.radioOption} ${!yoloAvailable ? styles.disabled : ''}`}
          title={!yoloAvailable ? 'YOLO detector not yet implemented' : ''}
        >
          <input
            type="radio"
            name="detector"
            value="yolo"
            checked={selectedDetector === 'yolo'}
            onChange={() => onDetectorChange('yolo')}
            disabled={!yoloAvailable}
          />
          <span className={styles.radioLabel}>
            <span className={styles.radioIcon}>🤖</span>
            YOLO
            {!yoloAvailable && <span className={styles.comingSoon}>Coming Soon</span>}
          </span>
          <span className={styles.radioDescription}>
            Deep learning-based detection (YOLOv8)
          </span>
        </label>
      </div>

      {!yoloAvailable && (
        <div className={styles.notice}>
          <strong>Note:</strong> Implement YOLO only after the classical detector works.
          This allows for experimental comparison between methods.
        </div>
      )}
    </div>
  );
}
