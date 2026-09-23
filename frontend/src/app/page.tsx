'use client';

// Fix: Added safe navigation for state property (2026-09-03 22:30)
import { useState, useEffect, useRef } from 'react';
import styles from './page.module.css';
import PresetButtons from './components/PresetButtons';
import DetectorSelector from './components/DetectorSelector';
import ExperimentHistory from './components/ExperimentHistory';
import MissionPanel from './components/MissionPanel';
import DatasetGenerator from './components/DatasetGenerator';
import { API_BASE, WS_BASE } from './config';
import {
  bboxToPercent,
  linkStateTone,
  patStateTone,
  resolveYoloPipelinePhase,
  sanitizeNonJsonNumbers,
  YOLO_ACTIVE_FLASH_MS,
  YOLO_WARMUP_TIMEOUT_MS,
} from './components/experimentFormat';

interface SimulationState {
  state: string;
  beacon_x: number;
  beacon_y: number;
  /** YOLO bounding box [x1, y1, x2, y2] in pixels; null for Classical. */
  bbox: [number, number, number, number] | null;
  /** Active detector as reported by backend telemetry. */
  activeDetector: 'classical' | 'yolo';
  /** Optical link block, stored verbatim; null before the first tick. */
  link: {
    received_power_w: number;
    pointing_loss_db: number | null;
    atmospheric_loss_db: number;
    link_margin_db: number;
    snr_db: number;
    link_state: string;
  } | null;
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
    bbox: null,
    activeDetector: 'classical',
    link: null,
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
  const [yoloAvailable, setYoloAvailable] = useState(false);
  // YOLO warmup/pipeline-fill UX (client-side only; no backend semantics).
  // Warming shows until the first backend-confirmed YOLO detection or the
  // warmup budget elapses; afterwards missing detections read as genuine
  // loss via the PAT badge. Never implies warmup misses are beacon loss.
  const [seenYoloDetection, setSeenYoloDetection] = useState(false);
  const [warmupExpired, setWarmupExpired] = useState(false);
  const [yoloActivatedFlash, setYoloActivatedFlash] = useState(false);
  const yoloWarmTimer = useRef<number | null>(null);
  const yoloFlashTimer = useRef<number | null>(null);
  const seenYoloDetectionRef = useRef(false);
  const warmingRef = useRef(false);
  const [connected, setConnected] = useState(false);
  const [connectAttempt, setConnectAttempt] = useState(0);
  const [connectNonce, setConnectNonce] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<number | null>(null);
  const unmountedRef = useRef(false);
  const attemptRef = useRef(0);
  const pendingDetector = useRef<'classical' | 'yolo' | null>(null);
  const selectedDetectorRef = useRef<'classical' | 'yolo'>('classical');
  // Mirror for socket callbacks (the connect effect mounts once).
  useEffect(() => {
    selectedDetectorRef.current = selectedDetector;
  });
  const clearYoloTimers = () => {
    if (yoloWarmTimer.current !== null) {
      window.clearTimeout(yoloWarmTimer.current);
      yoloWarmTimer.current = null;
    }
    if (yoloFlashTimer.current !== null) {
      window.clearTimeout(yoloFlashTimer.current);
      yoloFlashTimer.current = null;
    }
  };
  /** Settle warmup UX without touching detector selection or telemetry. */
  const settleYoloWarmup = () => {
    clearYoloTimers();
    warmingRef.current = false;
    setWarmupExpired(true);
    setYoloActivatedFlash(false);
  };
  /** Full reset to pre-selection state (switch-away, disconnect). */
  const resetYoloWarmup = () => {
    clearYoloTimers();
    seenYoloDetectionRef.current = false;
    warmingRef.current = false;
    setSeenYoloDetection(false);
    setWarmupExpired(false);
    setYoloActivatedFlash(false);
  };
  /** (Re)start YOLO warmup UX: banner until first nominal detection or budget. */
  const armYoloWarmup = () => {
    clearYoloTimers();
    seenYoloDetectionRef.current = false;
    warmingRef.current = true;
    setSeenYoloDetection(false);
    setWarmupExpired(false);
    setYoloActivatedFlash(false);
    yoloWarmTimer.current = window.setTimeout(() => {
      yoloWarmTimer.current = null;
      warmingRef.current = false;
      setWarmupExpired(true);
    }, YOLO_WARMUP_TIMEOUT_MS);
  };
  const disturbanceTimer = useRef<number | null>(null);
  const lastDisturbance = useRef<{ type: string; value: number } | null>(null);

  // Camera geometry mirrors backend VirtualCamera (1280x720).
  const CAM_W = 1280;
  const CAM_H = 720;

  /** Send only on an open socket; returns false instead of throwing. */
  const sendCommand = (payload: Record<string, unknown>): boolean => {
    const socket = wsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return false;
    try {
      socket.send(JSON.stringify(payload));
      return true;
    } catch {
      return false;
    }
  };

  // Detector availability comes from backend (/detector), not hardcoded.
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/detector`)
      .then((res) => (res.ok ? res.json() : null))
      .then((info) => {
        if (!cancelled && info) {
          setYoloAvailable(!!info.yolo_available);
          if (info.active === 'yolo' || info.active === 'classical') {
            setSelectedDetector(info.active);
          }
        }
      })
      .catch(() => {
        if (!cancelled) setYoloAvailable(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    unmountedRef.current = false;
    return () => {
      unmountedRef.current = true;
      if (disturbanceTimer.current !== null) {
        window.clearTimeout(disturbanceTimer.current);
        disturbanceTimer.current = null;
      }
      clearYoloTimers();
    };
  }, []);

  // WebSocket connection with exponential-backoff reconnect.
  useEffect(() => {
    let cancelled = false;
    let activeSocket: WebSocket | null = null;

    const scheduleReconnect = () => {
      if (cancelled || unmountedRef.current) return;
      const delay = Math.min(1000 * 2 ** attemptRef.current, 15000);
      attemptRef.current += 1;
      setConnectAttempt(attemptRef.current);
      reconnectTimer.current = window.setTimeout(() => {
        reconnectTimer.current = null;
        connect();
      }, delay);
    };

    const connect = () => {
      if (cancelled || unmountedRef.current) return;
      let socket: WebSocket;
      try {
        socket = new WebSocket(`${WS_BASE}/ws/simulation`);
      } catch {
        scheduleReconnect();
        return;
      }
      wsRef.current = socket;
      activeSocket = socket;

      socket.onopen = () => {
        if (cancelled) return;
        attemptRef.current = 0;
        setConnectAttempt(0);
        setConnected(true);
        // Refresh availability + sync active detector over the live socket.
        sendCommand({ command: 'get_detector' });
        // Re-arm YOLO warmup UX on (re)connect while YOLO is selected:
        // pipeline state after a drop is unknown until detections flow.
        if (selectedDetectorRef.current === 'yolo') {
          armYoloWarmup();
        }
      };

      socket.onmessage = (event) => {
        let data: {
          type?: string;
          command?: string;
          message?: unknown;
          detector?: { yolo_available?: boolean; active?: string; type?: string };
          detector_type?: string;
          pat?: { state?: string };
          detection?: { detected?: boolean; x?: number; y?: number; confidence?: number; detector_type?: string; bbox?: unknown };
          error?: { pixel?: number };
          performance?: { simulation_fps?: number; processing_fps?: number };
          gimbal?: { azimuth?: number; elevation?: number };
          fsoc?: { status?: string };
          link?: {
            received_power_w?: number;
            pointing_loss_db?: number | null;
            atmospheric_loss_db?: number;
            link_margin_db?: number;
            snr_db?: number;
            link_state?: string;
          } | null;
          simulation?: { running?: boolean; paused?: boolean };
          frame?: number;
          simulation_time?: number;
          image?: string;
        };
        try {
          // Sanitize Python-emitted Infinity/NaN tokens (outside strings)
          // so strict parsing survives out-of-range link values.
          data = JSON.parse(sanitizeNonJsonNumbers(event.data));
        } catch {
          return; // ignore malformed frames without killing the handler
        }
        if (!data || typeof data.type !== 'string') return;

        // Backend-driven detector availability (dynamic, not hardcoded).
        if (data.type === 'detector' && data.detector) {
          setYoloAvailable(!!data.detector.yolo_available);
          if (data.detector.active === 'yolo' || data.detector.active === 'classical') {
            pendingDetector.current = null;
            setSelectedDetector(data.detector.active);
          }
          return;
        }
        if (data.type === 'ack') {
          if (data.command === 'set_detector' && data.detector) {
            setYoloAvailable(!!data.detector.yolo_available);
            pendingDetector.current = null;
            if (data.detector_type === 'yolo' || data.detector_type === 'classical') {
              setSelectedDetector(data.detector_type);
            }
          }
          return;
        }
        if (data.type === 'error') {
          const msg = String(data.message ?? '');
          // Rejected optimistic detector switch: drop the pending change
          // and re-sync with backend truth instead of guessing.
          if (/detector|yolo|classical/i.test(msg) && pendingDetector.current) {
            pendingDetector.current = null;
            // Rejected switch is not warmup: settle so misses read normally.
            settleYoloWarmup();
            sendCommand({ command: 'get_detector' });
          } else {
            console.warn('Backend error:', msg);
          }
          return;
        }

        // Handle different message types
        if (data.type === 'telemetry') {
        const activeDetector = data.detector?.type || data.detection?.detector_type;
        if (activeDetector === 'yolo' || activeDetector === 'classical') {
          setSelectedDetector((prev) => (prev === activeDetector ? prev : activeDetector));
        }
        // Optical link block: stored verbatim (null before first tick).
        const rawLink = data.link;
        const link =
          rawLink !== null &&
          typeof rawLink === 'object' &&
          typeof rawLink.received_power_w === 'number' &&
          (typeof rawLink.pointing_loss_db === 'number' || rawLink.pointing_loss_db === null) &&
          typeof rawLink.atmospheric_loss_db === 'number' &&
          typeof rawLink.link_margin_db === 'number' &&
          typeof rawLink.snr_db === 'number' &&
          typeof rawLink.link_state === 'string'
            ? {
                received_power_w: rawLink.received_power_w,
                pointing_loss_db: rawLink.pointing_loss_db,
                atmospheric_loss_db: rawLink.atmospheric_loss_db,
                link_margin_db: rawLink.link_margin_db,
                snr_db: rawLink.snr_db,
                link_state: rawLink.link_state,
              }
            : null;
        // Backend sends nested structure - extract the values we need
        const rawBbox = data.detection?.bbox;
        const bbox: [number, number, number, number] | null =
          Array.isArray(rawBbox) &&
          rawBbox.length === 4 &&
          rawBbox.every((v) => typeof v === 'number' && Number.isFinite(v))
            ? [rawBbox[0], rawBbox[1], rawBbox[2], rawBbox[3]]
            : null;
        setSimState({
          state: data.pat?.state || 'IDLE',
          beacon_x: data.detection?.x || 640,
          beacon_y: data.detection?.y || 360,
          bbox,
          activeDetector: activeDetector === 'yolo' ? 'yolo' : 'classical',
          link,
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
        // First backend-confirmed nominal YOLO detection ends warming:
        // pipeline is filled, normal detection has started. The transient
        // confirmation is timer-dismissed; later misses read as loss.
        const teleActive = data.detector?.type || data.detection?.detector_type;
        const teleNominal =
          teleActive === 'yolo' &&
          !!data.detection?.detected &&
          (data.detection?.confidence || 0) > 0;
        if (teleNominal && !seenYoloDetectionRef.current) {
          seenYoloDetectionRef.current = true;
          setSeenYoloDetection(true);
          if (warmingRef.current) {
            warmingRef.current = false;
            if (yoloWarmTimer.current !== null) {
              window.clearTimeout(yoloWarmTimer.current);
              yoloWarmTimer.current = null;
            }
            setYoloActivatedFlash(true);
            if (yoloFlashTimer.current !== null) {
              window.clearTimeout(yoloFlashTimer.current);
            }
            yoloFlashTimer.current = window.setTimeout(() => {
              yoloFlashTimer.current = null;
              setYoloActivatedFlash(false);
            }, YOLO_ACTIVE_FLASH_MS);
          }
        }
      } else if (data.type === 'frame') {
        // Handle camera frame
        setCameraFrame({
          frame: data.frame ?? 0,
          simulation_time: data.simulation_time ?? 0,
          image: data.image ?? ''
        });
      }
    };

    // onerror fires alongside onclose; guard so we schedule one reconnect.
    let dropped = false;
    const handleDrop = () => {
      if (cancelled || dropped) return;
      dropped = true;
      setConnected(false);
      if (wsRef.current === socket) wsRef.current = null;
      // Drop warmup UX state; onopen re-arms it if YOLO is still selected.
      // Reconnect behavior itself is unchanged.
      resetYoloWarmup();
      scheduleReconnect();
    };
    socket.onerror = handleDrop;
    socket.onclose = handleDrop;
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer.current !== null) {
        window.clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      if (activeSocket) {
        activeSocket.onopen = null;
        activeSocket.onmessage = null;
        activeSocket.onerror = null;
        activeSocket.onclose = null;
        try {
          activeSocket.close();
        } catch {
          /* already closed */
        }
        if (wsRef.current === activeSocket) wsRef.current = null;
      }
    };
  }, [connectNonce]);

  const handleRetry = () => {
    attemptRef.current = 0;
    setConnectAttempt(0);
    setConnectNonce((n) => n + 1);
  };

  const handlePresetSelect = (preset: { turbulence: number; vibration: number; cameraMotion: number; noise: number }) => {
    setTurbulence(preset.turbulence);
    setVibration(preset.vibration);
    setCameraMotion(preset.cameraMotion);
    setNoise(preset.noise);

    // Send each disturbance separately (no-op while disconnected).
    sendCommand({ command: 'set_disturbance', name: 'turbulence', value: preset.turbulence });
    sendCommand({ command: 'set_disturbance', name: 'vibration', value: preset.vibration });
    sendCommand({ command: 'set_disturbance', name: 'camera_motion', value: preset.cameraMotion });
    sendCommand({ command: 'set_disturbance', name: 'sensor_noise', value: preset.noise });
  };

  const handleDetectorChange = (detector: 'classical' | 'yolo') => {
    const previous = selectedDetector;
    pendingDetector.current = detector;
    setSelectedDetector(detector);

    if (!sendCommand({ command: 'set_detector', detector_type: detector })) {
      // Socket down: revert the optimistic switch; it will resend on retry.
      pendingDetector.current = null;
      setSelectedDetector(previous);
      return;
    }
    if (detector === 'yolo') {
      // Model loads in the background; the loop stays live. Banner until
      // first nominal detection or budget — never a loss implication.
      armYoloWarmup();
    } else {
      resetYoloWarmup();
    }
  };

  const handleStart = () => {
    sendCommand({ command: 'start' });
  };

  const handlePause = () => {
    sendCommand({ command: 'pause' });
  };

  const handleResume = () => {
    sendCommand({ command: 'resume' });
  };

  const handleReset = () => {
    sendCommand({ command: 'reset' });
  };

  const handleDisturbanceChange = (type: string, value: number) => {
    // Debounce slider drags: UI updates instantly, backend gets the trailing value.
    lastDisturbance.current = { type, value };
    if (disturbanceTimer.current !== null) {
      window.clearTimeout(disturbanceTimer.current);
    }
    disturbanceTimer.current = window.setTimeout(() => {
      disturbanceTimer.current = null;
      const pending = lastDisturbance.current;
      if (pending) {
        sendCommand({ command: 'set_disturbance', name: pending.type, value: pending.value });
      }
    }, 150);
  };

  // YOLO overlay box derived from live telemetry via the tested helper;
  // Classical (bbox null) keeps the marker-only visualization unchanged.
  const yoloBox =
    simState.activeDetector === 'yolo' ? bboxToPercent(simState.bbox) : null;

  // YOLO pipeline phase for warmup UX (pure, tested): warming shows until
  // the first backend-confirmed nominal detection or the warmup budget;
  // settled states let PAT SEARCH read as genuine loss.
  const yoloPhase = resolveYoloPipelinePhase({
    selectedDetector,
    activeDetector: simState.activeDetector,
    loopLive: connected && simState.running && !simState.paused,
    seenYoloDetection,
    warmupExpired,
  });

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>FSOC PAT SIMULATOR</h1>
        <div className={styles.headerStatus}>
          {simState.running && !simState.paused && (
            <div className={styles.runStatus}>
              [+] RUNNING
            </div>
          )}
          {simState.paused && (
            <div className={styles.pauseStatus}>
              [-] PAUSED
            </div>
          )}
          <div className={`${styles.connectionStatus} ${connected ? styles.connected : styles.disconnected}`}>
            {connected
              ? '[+] CONNECTED'
              : connectAttempt > 0
                ? `[-] RECONNECTING… (${connectAttempt})`
                : '[x] DISCONNECTED'}
          </div>
          {!connected && (
            <button
              className={styles.btnResume}
              onClick={handleRetry}
              title="Reconnect to the simulation backend"
            >
              RETRY
            </button>
          )}
        </div>
      </header>

      <section className={styles.heroStrip} aria-label="Mission-critical readouts">
        <div className={styles.heroCell}>
          <span className={styles.heroLabel}>Pointing error</span>
          <span className={styles.heroValue}>
            {simState.error.toFixed(2)} <small>px</small>
          </span>
        </div>
        <div className={styles.heroCell}>
          <span className={styles.heroLabel}>Link margin</span>
          <span
            className={`${styles.heroValue} ${(() => {
              const tone = linkStateTone(simState.link?.link_state);
              return tone === 'nominal'
                ? styles.heroToneNominal
                : tone === 'marginal'
                  ? styles.heroToneMarginal
                  : tone === 'outage'
                    ? styles.heroToneOutage
                    : styles.heroToneNodata;
            })()}`}
          >
            {simState.link == null ? (
              <>— <small>dB</small></>
            ) : (
              <>
                {simState.link.link_margin_db.toFixed(2)} <small>dB</small>
              </>
            )}
          </span>
        </div>
        <div className={styles.heroCell}>
          <span className={styles.heroLabel}>PAT state</span>
          <span className={styles.heroBadgeRow}>
            <span className={`${styles.stateBadge} ${styles[patStateTone(simState.state)]}`}>
              {simState.state || 'IDLE'}
            </span>
          </span>
        </div>
        <div className={styles.heroCell}>
          <span className={styles.heroLabel}>Detector</span>
          <span className={styles.heroBadgeRow}>
            <span className={styles.heroDetector}>
              {simState.activeDetector === 'yolo' ? 'YOLO11' : 'CLASSICAL'}
            </span>
          </span>
        </div>
      </section>

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
                      left: `${(simState.beacon_x / CAM_W) * 100}%`,
                      top: `${(simState.beacon_y / CAM_H) * 100}%`
                    }}
                  />

                  {/* Overlay: YOLO bounding box + confidence (YOLO only) */}
                  {yoloBox && (
                    <>
                      <div
                        className={styles.yoloBbox}
                        style={{
                          left: `${yoloBox.leftPct}%`,
                          top: `${yoloBox.topPct}%`,
                          width: `${yoloBox.widthPct}%`,
                          height: `${yoloBox.heightPct}%`
                        }}
                      />
                      <div
                        className={styles.yoloLabel}
                        style={{
                          left: `${yoloBox.leftPct}%`,
                          top: `${yoloBox.topPct}%`
                        }}
                      >
                        YOLO {(simState.confidence * 100).toFixed(0)}%
                      </div>
                    </>
                  )}

                  {/* Overlay: Error vector from detection to center */}
                  <svg className={styles.errorVector}>
                    <line
                      x1={`${(simState.beacon_x / CAM_W) * 100}%`}
                      y1={`${(simState.beacon_y / CAM_H) * 100}%`}
                      x2="50%"
                      y2="50%"
                      stroke="#ff3b30"
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
            <h2>PAT STATE</h2>
            <div className={`${styles.stateBadge} ${styles[patStateTone(simState.state)]}`}>
              {simState.state || 'IDLE'}
            </div>
          </div>

          {yoloPhase === 'warming' && (
            <div className={styles.statusCard} role="status" aria-live="polite">
              <h2>YOLO PIPELINE</h2>
              <div className={styles.warmupBadge}>WARMING UP</div>
              <div className={styles.warmupHint}>
                Model loading in background — control loop live
                {simState.processing_fps > 0
                  ? ` @ ${simState.processing_fps.toFixed(0)} Hz`
                  : ''}
                . Missing detections are expected while the pipeline
                fills, not beacon loss.
              </div>
            </div>
          )}
          {yoloActivatedFlash && selectedDetector === 'yolo' && (
            <div className={styles.statusCard} role="status">
              <h2>YOLO PIPELINE</h2>
              <div className={styles.activeBadge}>
                ACTIVE — nominal detections
              </div>
            </div>
          )}

          <div className={styles.statusCard}>
            <h2>OPTICAL LINK</h2>
            {(() => {
              const tone = linkStateTone(simState.link?.link_state);
              const toneClass =
                tone === 'nominal'
                  ? styles.linkNominal
                  : tone === 'marginal'
                    ? styles.linkMarginal
                    : tone === 'outage'
                      ? styles.linkOutage
                      : styles.linkNodata;
              const label =
                simState.link == null ? 'NO DATA' : simState.link.link_state;
              return (
                <div className={`${styles.linkBadge} ${toneClass}`}>
                  <span className={styles.linkDot} />
                  {label}
                </div>
              );
            })()}
            <div className={styles.readoutTable}>
              <div className={styles.readoutRow}>
                <span className={styles.readoutLabel}>SNR</span>
                <span className={styles.readoutValue}>
                  {simState.link == null ? '—' : `${simState.link.snr_db.toFixed(2)} dB`}
                </span>
              </div>
              <div className={styles.readoutRow}>
                <span className={styles.readoutLabel}>Link Margin</span>
                <span className={styles.readoutValue}>
                  {simState.link == null ? '—' : `${simState.link.link_margin_db.toFixed(2)} dB`}
                </span>
              </div>
              <div className={styles.readoutRow}>
                <span className={styles.readoutLabel}>Received Power</span>
                <span className={styles.readoutValue}>
                  {simState.link == null ? '—' : `${simState.link.received_power_w.toExponential(2)} W`}
                </span>
              </div>
              <div className={styles.readoutRow}>
                <span className={styles.readoutLabel}>Pointing Loss</span>
                <span className={styles.readoutValue}>
                  {simState.link == null || simState.link.pointing_loss_db == null
                    ? '—'
                    : `${simState.link.pointing_loss_db.toFixed(2)} dB`}
                </span>
              </div>
              <div className={styles.readoutRow}>
                <span className={styles.readoutLabel}>Atmos Loss</span>
                <span className={styles.readoutValue}>
                  {simState.link == null ? '—' : `${simState.link.atmospheric_loss_db.toFixed(2)} dB`}
                </span>
              </div>
            </div>
          </div>

          <div className={styles.readoutTable}>
            <div className={styles.readoutRow}>
              <span className={styles.readoutLabel}>Beacon X</span>
              <span className={styles.readoutValue}>{simState.beacon_x.toFixed(1)} px</span>
            </div>

            <div className={styles.readoutRow}>
              <span className={styles.readoutLabel}>Beacon Y</span>
              <span className={styles.readoutValue}>{simState.beacon_y.toFixed(1)} px</span>
            </div>

            <div className={styles.readoutRow}>
              <span className={styles.readoutLabel}>Error</span>
              <span className={styles.readoutValue}>{simState.error.toFixed(2)} px</span>
            </div>

            <div className={styles.readoutRow}>
              <span className={styles.readoutLabel}>Confidence</span>
              <span className={styles.readoutValue}>{(simState.confidence * 100).toFixed(1)}%</span>
            </div>

            <div className={styles.readoutRow}>
              <span className={styles.readoutLabel}>Azimuth</span>
              <span className={styles.readoutValue}>{simState.azimuth.toFixed(2)}°</span>
            </div>

            <div className={styles.readoutRow}>
              <span className={styles.readoutLabel}>Elevation</span>
              <span className={styles.readoutValue}>{simState.elevation.toFixed(2)}°</span>
            </div>

            <div className={styles.readoutRow}>
              <span className={styles.readoutLabel}>Sim FPS</span>
              <span className={styles.readoutValue}>{simState.simulation_fps.toFixed(0)} Hz</span>
            </div>

            <div className={styles.readoutRow}>
              <span className={styles.readoutLabel}>Proc FPS</span>
              <span className={styles.readoutValue}>{simState.processing_fps.toFixed(1)}</span>
            </div>

            <div className={styles.readoutRow}>
              <span className={styles.readoutLabel}>Link</span>
              <span className={`${styles.readoutValue} ${styles.linkStatus}`}>
                {simState.link_status}
              </span>
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
            yoloAvailable={yoloAvailable}
            yoloWarming={yoloPhase === 'warming'}
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

      <ExperimentHistory yoloAvailable={yoloAvailable} />

      <MissionPanel detector={selectedDetector} />
    </div>
  );
}
