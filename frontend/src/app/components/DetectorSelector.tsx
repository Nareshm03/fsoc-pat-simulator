'use client';

import styles from './DetectorSelector.module.css';

interface DetectorSelectorProps {
  selectedDetector: 'classical' | 'yolo';
  onDetectorChange: (detector: 'classical' | 'yolo') => void;
  yoloAvailable?: boolean;
  /** YOLO pipeline filling (model load + first observations); tag only,
   *  switching away stays enabled as the escape hatch. */
  yoloWarming?: boolean;
}

export default function DetectorSelector({
  selectedDetector,
  onDetectorChange,
  yoloAvailable = false,
  yoloWarming = false
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
            <span className={styles.radioIcon}>[+]</span>
            Classical CV
          </span>
          <span className={styles.radioDescription}>
            OpenCV-based detection (threshold + contours)
          </span>
        </label>

        <label
          className={`${styles.radioOption} ${!yoloAvailable ? styles.disabled : ''}`}
          title={!yoloAvailable ? 'YOLO detector unavailable (backend offline or model missing)' : ''}
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
            <span className={styles.radioIcon}>[+]</span>
            YOLO
            {!yoloAvailable && <span className={styles.comingSoon}>Unavailable</span>}
            {yoloAvailable && yoloWarming && (
              <span className={styles.comingSoon} role="status">Warming…</span>
            )}
          </span>
          <span className={styles.radioDescription}>
            Deep learning-based detection (YOLO11n)
          </span>
        </label>
      </div>

      {!yoloAvailable && (
        <div className={styles.notice}>
          <strong>Note:</strong> YOLO is currently unavailable. Check that the
          backend is running and the model file is present, or continue with
          Classical CV.
        </div>
      )}
    </div>
  );
}
