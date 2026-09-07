'use client';

import styles from './PresetButtons.module.css';

interface DisturbancePreset {
  turbulence: number;
  vibration: number;
  cameraMotion: number;
  noise: number;
}

interface PresetButtonsProps {
  onPresetSelect: (preset: DisturbancePreset) => void;
}

const PRESETS: Record<string, DisturbancePreset> = {
  IDEAL: {
    turbulence: 0.0,
    vibration: 0.0,
    cameraMotion: 0.0,
    noise: 0.1,
  },
  'CAMERA MOTION': {
    turbulence: 0.2,
    vibration: 0.1,
    cameraMotion: 2.0,
    noise: 0.3,
  },
  VIBRATION: {
    turbulence: 0.3,
    vibration: 2.0,
    cameraMotion: 0.2,
    noise: 0.4,
  },
  TURBULENCE: {
    turbulence: 2.0,
    vibration: 0.2,
    cameraMotion: 0.3,
    noise: 0.5,
  },
  'HIGH NOISE': {
    turbulence: 0.5,
    vibration: 0.5,
    cameraMotion: 0.5,
    noise: 2.0,
  },
  'WORST CASE': {
    turbulence: 2.0,
    vibration: 2.0,
    cameraMotion: 1.5,
    noise: 2.0,
  },
};

export default function PresetButtons({ onPresetSelect }: PresetButtonsProps) {
  return (
    <div className={styles.presetContainer}>
      <div className={styles.presetLabel}>EXPERIMENT PRESETS</div>
      <div className={styles.presetButtons}>
        {Object.entries(PRESETS).map(([name, preset]) => (
          <button
            key={name}
            className={styles.presetBtn}
            onClick={() => onPresetSelect(preset)}
            title={`Turbulence: ${preset.turbulence}, Vibration: ${preset.vibration}, Camera: ${preset.cameraMotion}, Noise: ${preset.noise}`}
          >
            {name}
          </button>
        ))}
      </div>
    </div>
  );
}
