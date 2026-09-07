'use client';

// Fix: Added safe navigation for state property (2026-09-03 22:30)
import { useState, useEffect } from 'react';
import styles from './page.module.css';
import PresetButtons from './components/PresetButtons';
import DetectorSelector from './components/DetectorSelector';
import ExperimentHistory from './components/ExperimentHistory';
import DatasetGenerator from './components/DatasetGenerator';

interface SimulationState {
  state: string;
  beacon_x: number;
  beacon_y: number;
  error: number;
  simulation_fps: number;
  processing_fps: number;
  azimuth: number;
  elevation: number;
  confidence: number;
  link_status: string;
  running: boolean;
  paused: boolean;
}

interface CameraFrame {
  frame: number;
  simulation_time: number;
  image: string; // base64 JPEG
}

export default function Home() {
  const [simState, setSimState] = useState<SimulationState>({
    state: 'IDLE',
    beacon_x: 640,
    beacon_y: 360,
    error: 0,
    simulation_fps: 30,
    processing_fps: 0,
    azimuth: 0,
    elevation: 0,
    confidence: 0,
    link_status: 'DISCONNECTED',
    running: false,
    paused: false
  });

  const [cameraFrame, setCameraFrame] = useState<CameraFrame | null>(null);

  const [turbulence, setTurbulence] = useState(1.0);
  const [vibration, setVibration] = useState(1.0);
  const [cameraMotion, setCameraMotion] = useState(1.0);
  const [noise, setNoise] = useState(1.0);
  const [selectedDetector, setSelectedDetector] = useState<'classical' | 'yolo'>('classical');
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  // WebSocket connection
  useEffect(() => {
    const websocket = new WebSocket('ws://localhost:8000/ws/simulation');
    
    websocket.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // Handle different message types
      if (data.type === 'telemetry') {
        // Backend sends nested structure - extract the values we need
        setSimState({
          state: data.pat?.state || 'IDLE',
          beacon_x: data.detection?.x || 640,
          beacon_y: data.detection?.y || 360,
          error: data.error?.pixel || 0,
          simulation_fps: data.performance?.simulation_fps || 30,
          processing_fps: data.performance?.processing_fps || 0,
          azimuth: data.gimbal?.azimuth || 0,
          elevation: data.gimbal?.elevation || 0,
          confidence: data.detection?.confidence || 0,
          link_status: data.fsoc?.status || 'LOST',
          running: data.simulation?.running || false,
          paused: data.simulation?.paused || false
        });
      } else if (data.type === 'frame') {
        // Handle camera frame
        setCameraFrame({
          frame: data.frame,
          simulation_time: data.simulation_time,
          image: data.image
        });
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
    };

    setWs(websocket);

    return () => {
      websocket.close();
    };
  }, []);

  const handlePresetSelect = (preset: { turbulence: number; vibration: number; cameraMotion: number; noise: number }) => {
    console.log('[PRESET] Selected preset while state:', simState.paused ? 'PAUSED' : simState.running ? 'RUNNING' : 'IDLE', preset);
    setTurbulence(preset.turbulence);
    setVibration(preset.vibration);
    setCameraMotion(preset.cameraMotion);
    setNoise(preset.noise);

    if (ws && connected) {
      // Send each disturbance separately
      ws.send(JSON.stringify({
        command: 'set_disturbance',
        name: 'turbulence',
        value: preset.turbulence
      }));
      ws.send(JSON.stringify({
        command: 'set_disturbance',
        name: 'vibration',
        value: preset.vibration
      }));
      ws.send(JSON.stringify({
        command: 'set_disturbance',
        name: 'camera_motion',
        value: preset.cameraMotion
      }));
      ws.send(JSON.stringify({
        command: 'set_disturbance',
        name: 'sensor_noise',
        value: preset.noise
      }));
      console.log('[PRESET] Commands sent to backend');
    }
  };

  const handleDetectorChange = (detector: 'classical' | 'yolo') => {
    setSelectedDetector(detector);

    if (ws && connected) {
      ws.send(JSON.stringify({
        command: 'set_detector',
        detector_type: detector
      }));
    }
  };

  const handleStart = () => {
    console.log('[CONTROL] START clicked');
    if (ws && connected) {
      ws.send(JSON.stringify({ command: 'start' }));
    }
  };

  const handlePause = () => {
    console.log('[CONTROL] PAUSE clicked');
    if (ws && connected) {
      ws.send(JSON.stringify({ command: 'pause' }));
    }
  };

  const handleResume = () => {
    console.log('[CONTROL] RESUME clicked');
    if (ws && connected) {
      ws.send(JSON.stringify({ command: 'resume' }));
    }
  };

  const handleReset = () => {
    console.log('[CONTROL] RESET clicked from state:', simState.paused ? 'PAUSED' : simState.running ? 'RUNNING' : 'IDLE');
    if (ws && connected) {
      ws.send(JSON.stringify({ command: 'reset' }));
    }
  };

  const handleDisturbanceChange = (type: string, value: number) => {
    console.log(`[DISTURBANCE] ${type} changed to ${value} while state:`, simState.paused ? 'PAUSED' : simState.running ? 'RUNNING' : 'IDLE');
    if (ws && connected) {
      ws.send(JSON.stringify({
        command: 'set_disturbance',
        name: type,
        value: value
      }));
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>FSOC PAT SIMULATOR</h1>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          {simState.running && !simState.paused && (
            <div style={{ color: '#00ff88', fontSize: '0.875rem', fontWeight: 600 }}>
              ▶ RUNNING
            </div>
          )}
          {simState.paused && (
            <div style={{ color: '#ffaa00', fontSize: '0.875rem', fontWeight: 600 }}>
              ⏸ PAUSED
            </div>
          )}
          <div className={`${styles.connectionStatus} ${connected ? styles.connected : styles.disconnected}`}>
            {connected ? '● CONNECTED' : '○ DISCONNECTED'}
          </div>
        </div>
      </header>

      <div className={styles.mainContent}>
        <div className={styles.leftPanel}>
          <div className={styles.cameraView}>
            <div className={styles.cameraFrame}>
              {/* Real camera frame from backend */}
              {cameraFrame?.image ? (
                <img 
                  src={`data:image/jpeg;base64,${cameraFrame.image}`}
                  alt="Camera Feed"
                  className={styles.cameraImage}
                />
              ) : (
                <div className={styles.noStream}>
                  {connected ? 'WAITING FOR STREAM...' : 'CAMERA OFFLINE'}
                </div>
              )}
              
              {/* Overlay: Crosshair at center (always visible when frame exists) */}
              {cameraFrame?.image && (
                <div className={styles.crosshair}>
                  <div className={styles.crosshairH} />
                  <div className={styles.crosshairV} />
                </div>
              )}
              
              {/* Overlay: Detection marker (only when detected) */}
              {cameraFrame?.image && simState.confidence > 0 && (
                <>
                  <div 
                    className={styles.detectionMarker}
                    style={{
                      left: `${(simState.beacon_x / 1280) * 100}%`,
                      top: `${(simState.beacon_y / 720) * 100}%`
                    }}
                  />
                  
                  {/* Overlay: Error vector from detection to center */}
                  <svg className={styles.errorVector}>
                    <line
                      x1={`${(simState.beacon_x / 1280) * 100}%`}
                      y1={`${(simState.beacon_y / 720) * 100}%`}
                      x2="50%"
                      y2="50%"
                      stroke="#ff4444"
                      strokeWidth="2"
                      strokeDasharray="5,5"
                    />
                  </svg>
                </>
              )}
            </div>
            <div className={styles.cameraLabel}>
              VIRTUAL CAMERA {cameraFrame && `(Frame #${cameraFrame.frame})`}
            </div>
          </div>
        </div>

        <div className={styles.rightPanel}>
          <div className={styles.statusCard}>
            <h2>SIMULATION STATE</h2>
            <div className={`${styles.stateBadge} ${(() => {
              if (simState.paused) return styles.paused;
              if (simState.running) return styles.running;
              return styles.idle;
            })()}`}>
              {simState.paused ? 'PAUSED' : simState.running ? 'RUNNING' : 'IDLE'}
            </div>
          </div>

          <div className={styles.statusCard}>
            <h2>PAT STATE</h2>
            <div className={`${styles.stateBadge} ${(() => {
              const stateStr = simState.state || 'IDLE';
              const stateLower = stateStr.toLowerCase();
              return styles[stateLower] || styles.idle;
            })()}`}>
              {simState.state || 'IDLE'}
            </div>
          </div>

          <div className={styles.metricsGrid}>
            <div className={styles.metric}>
              <div className={styles.metricLabel}>Beacon X</div>
              <div className={styles.metricValue}>{simState.beacon_x.toFixed(1)} px</div>
            </div>

            <div className={styles.metric}>
              <div className={styles.metricLabel}>Beacon Y</div>
              <div className={styles.metricValue}>{simState.beacon_y.toFixed(1)} px</div>
            </div>

            <div className={styles.metric}>
              <div className={styles.metricLabel}>Error</div>
              <div className={styles.metricValue}>{simState.error.toFixed(2)} px</div>
            </div>

            <div className={styles.metric}>
              <div className={styles.metricLabel}>Sim FPS</div>
              <div className={styles.metricValue}>{simState.simulation_fps.toFixed(0)} Hz</div>
            </div>

            <div className={styles.metric}>
              <div className={styles.metricLabel}>Proc FPS</div>
              <div className={styles.metricValue}>{simState.processing_fps.toFixed(1)}</div>
            </div>

            <div className={styles.metric}>
              <div className={styles.metricLabel}>Azimuth</div>
              <div className={styles.metricValue}>{simState.azimuth.toFixed(2)}°</div>
            </div>

            <div className={styles.metric}>
              <div className={styles.metricLabel}>Elevation</div>
              <div className={styles.metricValue}>{simState.elevation.toFixed(2)}°</div>
            </div>

            <div className={styles.metric}>
              <div className={styles.metricLabel}>Confidence</div>
              <div className={styles.metricValue}>{(simState.confidence * 100).toFixed(1)}%</div>
            </div>

            <div className={styles.metric}>
              <div className={styles.metricLabel}>Link</div>
              <div className={`${styles.metricValue} ${styles.linkStatus}`}>
                {simState.link_status}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.controlPanel}>
        <PresetButtons onPresetSelect={handlePresetSelect} />

        <div className={styles.sliderGroup}>
          <label>
            Turbulence
            <input 
              type="range" 
              min="0" 
              max="2" 
              step="0.1" 
              value={turbulence}
              onChange={(e) => {
                const value = parseFloat(e.target.value);
                setTurbulence(value);
                handleDisturbanceChange('turbulence', value);
              }}
            />
            <span>{turbulence.toFixed(1)}</span>
          </label>

          <label>
            Vibration
            <input 
              type="range" 
              min="0" 
              max="2" 
              step="0.1" 
              value={vibration}
              onChange={(e) => {
                const value = parseFloat(e.target.value);
                setVibration(value);
                handleDisturbanceChange('vibration', value);
              }}
            />
            <span>{vibration.toFixed(1)}</span>
          </label>

          <label>
            Camera Motion
            <input 
              type="range" 
              min="0" 
              max="2" 
              step="0.1" 
              value={cameraMotion}
              onChange={(e) => {
                const value = parseFloat(e.target.value);
                setCameraMotion(value);
                handleDisturbanceChange('camera_motion', value);
              }}
            />
            <span>{cameraMotion.toFixed(1)}</span>
          </label>

          <label>
            Noise
            <input 
              type="range" 
              min="0" 
              max="2" 
              step="0.1" 
              value={noise}
              onChange={(e) => {
                const value = parseFloat(e.target.value);
                setNoise(value);
                handleDisturbanceChange('sensor_noise', value);
              }}
            />
            <span>{noise.toFixed(1)}</span>
          </label>
        </div>

        <div className={styles.bottomRow}>
          <DetectorSelector 
            selectedDetector={selectedDetector}
            onDetectorChange={handleDetectorChange}
            yoloAvailable={false}
          />

          <div className={styles.buttonGroup}>
            <button 
              className={styles.btnStart} 
              onClick={handleStart}
              disabled={simState.running}
              title="Start new simulation"
            >
              START
            </button>
            <button 
              className={styles.btnPause} 
              onClick={handlePause}
              disabled={!simState.running || simState.paused}
              title="Pause simulation"
            >
              PAUSE
            </button>
            <button 
              className={styles.btnResume} 
              onClick={handleResume}
              disabled={!simState.paused}
              title="Resume paused simulation"
            >
              RESUME
            </button>
            <button 
              className={styles.btnReset} 
              onClick={handleReset}
              title="Reset simulation to idle"
            >
              RESET
            </button>
          </div>
        </div>
      </div>

      <DatasetGenerator connected={connected} />

      <ExperimentHistory />
    </div>
  );
}
