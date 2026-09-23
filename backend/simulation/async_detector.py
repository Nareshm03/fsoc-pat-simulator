"""Bounded asynchronous detection wrapper: newest frame only, timestamped.

Decouples the 30 Hz simulation/control tick from YOLO inference latency
(~70-100 ms steady-state at imgsz=416). A single-flight worker always
infers the newest submitted frame; per-tick decisions come from the
newest *completed* observation, gated by an explicit age bound.

Freshness contract (this is the anti-faking core, read carefully):

- ``detected`` is True only while the newest completed observation is a
  *positive* detection no older than ``max_age_s`` simulation seconds.
  The reported age (``last_age_ticks`` frame count, kept for telemetry)
  and freshness (``last_fresh``) travel with every decision, so a coasted
  tick is never mistaken for a fresh detection.
- Between real observations the reported position comes from a
  complementary world track, never a static copy: the track is predicted
  forward every tick with the smoothed measured rate, and each strictly
  newer real fix corrects it with position_gain (plus exact gimbal
  re-projection). Spreading arrival corrections over ticks keeps the
  reported stream continuous so the controller feedforward is not
  kicked by per-tick jumps. No predicted detections are invented beyond
  the age bound; out-of-frame predictions and expired observations
  report undetected, preserving genuine loss timing and the existing
  SEARCH -> REACQUIRE logic untouched.
- A fresh negative observation (real miss / blank frame) reports
  undetected immediately -- loss is never masked by coasting.

Threading: ``submit``/``pump``/``detect``/``begin_tick`` are called from
the event-loop thread; inference runs in worker threads via
``asyncio.to_thread``. All shared slots are guarded by one lock.
``detect()`` itself never blocks: it only inspects completed results.

Lifecycle: ``reset_generation()`` (detector switch, sim reset) retires
all in-flight and completed observations; completions tagged with a
stale generation are discarded, never applied. ``shutdown()`` cancels
the worker so no threads/tasks outlive the simulation.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum age, in simulation seconds, of a positive observation that may
# still back a tick decision. Pipeline bound, NOT a PAT threshold
# (lock/unlock/reacquire thresholds are untouched). Default 0.4 s == 12
# ticks at the 30 Hz live cadence, preserving the previous behavior
# there exactly; unlike a tick count, the timeout in seconds stays
# invariant when dt changes (12 ticks would be 0.2 s at 60 Hz but 0.6 s
# at 20 Hz). Rationale: completions can only be harvested on a later
# tick than their capture frame, so baseline age is already ~0.1 s at
# 80 ms inference, and measured ages reach 0.4 s under jitter (expiry
# flips ticks to undetected, firing REACQUIRE+raster+FF-reset and
# breaking lock retention). 0.4 s keeps continuity with margin; a fresh
# negative observation still reports genuine loss immediately, so loss
# latency stays ~1 inference period (no masking).
MAX_OBSERVATION_AGE_S = 0.4
# Telemetry tick bound at the 30 Hz live cadence (0.4 s == 12 ticks).
# Expiry itself is time-based (max_age_s); this constant only bounds the
# frame-count age reported via last_age_ticks / observation_age_ticks.
MAX_OBSERVATION_AGE_TICKS = 12

# Tolerance (s) for simulation-time comparisons. Covers float
# accumulation dust while staying far below any real tick spacing.
_TIME_EPS = 1e-9


class AsyncDetector:
    """Newest-frame-only async wrapper around a synchronous detector."""

    def __init__(self, detector: Any, target: Any,
                 max_age_s: float = MAX_OBSERVATION_AGE_S,
                 rate_alpha: float = 0.3,
                 position_gain: float = 0.6,
                 harvest_interval_s: float = 0.1):
        if (not isinstance(max_age_s, (int, float))
                or not math.isfinite(max_age_s)
                or max_age_s < 0):
            raise ValueError("max_age_s must be a non-negative finite number")
        if not 0.0 < rate_alpha <= 1.0:
            raise ValueError("rate_alpha must be in (0, 1]")
        if not 0.0 < position_gain <= 1.0:
            raise ValueError("position_gain must be in (0, 1]")
        if (not isinstance(harvest_interval_s, (int, float))
                or not math.isfinite(harvest_interval_s)
                or harvest_interval_s <= 0):
            raise ValueError("harvest_interval_s must be a positive finite number")
        self.detector = detector
        self._target = target
        self.max_age_s = float(max_age_s)
        # EWMA weight for the world-rate estimate across real
        # observations. The two-point rate aliases the 6-7 Hz vibration
        # component (observations arrive at ~12 Hz); smoothing rejects
        # it while genuine target motion (<=0.5 rad/s bandwidth) passes
        # with negligible lag. Measured sweep (disturbed, seed 7):
        # alpha 0.3 locks (frac 0.27), 0.5 locks (0.13), 0.7 never
        # locks -- heavier raw-rate weighting chases aliased vibration.
        self.rate_alpha = rate_alpha
        self._rate_waz = 0.0
        self._rate_wel = 0.0
        # Complementary world track: predicted forward every tick with
        # the smoothed rate, corrected toward each new real fix with
        # position_gain. This keeps the reported stream continuous
        # across arrivals (no per-tick jumps for the controller FF to
        # chase) while every correction comes from a real observation.
        # Measured sweep (disturbed, seed 7): K=0.6 locks (frac 0.27),
        # K=0.4 locks (0.22), K=0.85/1.0 barely lock (0.03/0.02) --
        # near-full jumps reintroduce FF spiking. _track is dict(waz,
        # wel, t) plus basis fn in _track_fn.
        self.position_gain = position_gain
        # Harvest regularization (jitter root-cause fix): completed
        # positive observations are applied to _latest only when the
        # current tick reaches the next point of a ~0.1 s simulation-time
        # grid (multiples of harvest_interval_s from the sim origin),
        # coasting otherwise with honest age. The grid is absolute, so
        # changing dt cannot accumulate drift: at dt=1/30 harvests land
        # every 3rd tick exactly as before, at dt=1/60 every 6th, at
        # dt=1/20 every 2nd. Fresh negatives (real misses) still apply
        # immediately so genuine loss latency stays ~1 inference period
        # (never masked). Pipeline stays full via a one-slot _held buffer
        # (no idle): completions clear the task slot immediately,
        # launching the next inference, while _latest application waits
        # for the grid. No wall-clock scheduling is involved; only the
        # caller-supplied simulation time drives harvesting.
        self.harvest_interval_s = float(harvest_interval_s)
        self._held: Optional[Dict[str, Any]] = None
        # inv: _next_harvest_time is a grid multiple k*interval (k >= 1,
        # within float ulp) strictly greater than the last tick time at
        # which a harvest was applied (+eps tolerance for float dust).
        self._next_harvest_time: float = float(harvest_interval_s)
        self._track = None
        self._track_fn: Optional[int] = None
        # Reentrant: _track_resolve holds the lock while _note_world
        # re-acquires it during fresh-observation folding.
        self._lock = threading.RLock()
        self._generation = 0
        self._pending: Optional[Dict[str, Any]] = None
        self._task: Optional[asyncio.Task] = None
        self._latest: Optional[Dict[str, Any]] = None
        # Last two positive *real* observations, world-fixed:
        # (world_az, world_el, sim_time) of last positive REAL obs.
        # Rate source for propagation. _noted_fn makes folding idempotent
        # so repeated coast ticks on one observation never shift history.
        self._prev_world: Optional[Tuple[float, float, float]] = None
        self._last_world: Optional[Tuple[float, float, float]] = None
        self._noted_fn: Optional[int] = None
        # Current-tick context, set by begin_tick before detect().
        self._tick_sim_time = 0.0
        self._tick_frame = 0
        self._tick_gimbal_az = 0.0
        self._tick_gimbal_el = 0.0
        # Last decision provenance (for telemetry; same-thread access).
        self.last_age_ticks: Optional[int] = None
        self.last_fresh: bool = False
        # Counters (telemetry/tests): submissions vs completed inferences.
        self.frames_submitted = 0
        self.inferences_started = 0
        self.inferences_completed = 0

    # ------------------------------------------------------------------
    # Loop-side driver API (called from the event-loop thread each tick).
    # ------------------------------------------------------------------
    def begin_tick(self, sim_time: float, frame_number: int,
                   gimbal_az_rad: float, gimbal_el_rad: float) -> None:
        """Record current-tick context for the upcoming detect() call."""
        with self._lock:
            self._tick_sim_time = sim_time
            self._tick_frame = frame_number
            self._tick_gimbal_az = gimbal_az_rad
            self._tick_gimbal_el = gimbal_el_rad

    # Tolerance (s) for simulation-time comparisons. Covers float
    # accumulation dust (e.g. 3 additions of 1/30 fall ~1e-17 below 0.1)
    # while staying far below any real tick spacing.
    _HARVEST_EPS = _TIME_EPS

    def _harvest_due(self, cur_t: float) -> bool:
        """True when the current tick has reached the harvest grid."""
        return self._next_harvest_time <= cur_t + self._HARVEST_EPS

    def _advance_harvest_grid(self, cur_t: float) -> None:
        """Move the grid past the current tick in one closed-form step.

        inv: on entry _next_harvest_time is on-grid.
        post: _next_harvest_time is the smallest grid multiple strictly
          greater than cur_t (+eps). A tick that skips several grid
          points advances past all of them at once, so neither large dt
          steps nor dt changes can accumulate drift (error stays at float
          ulp level, far below _HARVEST_EPS).
        term: single arithmetic computation, no loop.
        """
        # inv: grid membership is preserved by construction (init/reset
        # anchor it; every advance adds whole multiples of the interval).
        if self._next_harvest_time <= cur_t + self._HARVEST_EPS:
            skipped = math.floor(
                (cur_t + self._HARVEST_EPS - self._next_harvest_time)
                / self.harvest_interval_s)
            self._next_harvest_time += (skipped + 1) * self.harvest_interval_s

    def pump(self) -> None:
        """Harvest a finished inference and launch the next, never blocking.

        Must be called from the event loop. At most one inference is ever
        in flight; a newer pending frame simply replaces the older one.
        Positive completions wait in _held for the next ~0.1 s
        simulation-time grid point; fresh negatives apply immediately so
        genuine loss is never masked. The task slot clears on completion
        (into _held/_latest), so the pipeline never idles.
        """
        with self._lock:
            cur_t = self._tick_sim_time
            if self._task is not None and self._task.done():
                try:
                    obs = self._task.result()
                except Exception:
                    logger.warning("Async detection worker failed",
                                   exc_info=True)
                    obs = None
                self._task = None
                if obs is not None and obs["generation"] == self._generation:
                    self.inferences_completed += 1
                    det = obs.get("detection")
                    if det is None:
                        # Fresh negative: genuine loss path, never held.
                        self._latest = obs
                        self._held = None
                    elif self._harvest_due(cur_t):
                        self._latest = obs
                        # A newer positive supersedes any older held one.
                        if (self._held is not None and self._held["frame_number"]
                                <= obs["frame_number"]):
                            self._held = None
                        self._advance_harvest_grid(cur_t)
                    else:
                        # Hold newest positive for the next grid point.
                        if (self._held is None or obs["frame_number"]
                                >= self._held["frame_number"]):
                            self._held = obs
            # Apply a held positive on grid points (newest wins by frame
            # number, so a newer real miss in _latest is never masked by
            # older held data).
            if self._held is not None and self._harvest_due(cur_t):
                held = self._held
                cur = self._latest
                if cur is None or held["frame_number"] > cur["frame_number"]:
                    self._latest = held
                    self._held = None
                    self._advance_harvest_grid(cur_t)
                elif held["frame_number"] <= cur["frame_number"]:
                    self._held = None
            if self._task is None and self._pending is not None:
                meta = self._pending
                self._pending = None
                self.inferences_started += 1
                self._task = asyncio.get_running_loop().create_task(
                    self._run_one(meta)
                )

    async def _run_one(self, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Infer one frame in a worker thread; return timestamped result."""
        try:
            detection = await asyncio.to_thread(
                self.detector.detect, meta["frame"]
            )
        except Exception:
            logger.warning("Async detection inference failed", exc_info=True)
            return None
        return {
            "detection": detection,
            "sim_time": meta["sim_time"],
            "frame_number": meta["frame_number"],
            "gimbal_az": meta["gimbal_az"],
            "gimbal_el": meta["gimbal_el"],
            "generation": meta["generation"],
        }

    def reset_generation(self) -> None:
        """Retire all pending/completed observations (switch/reset).

        In-flight work is left to finish harmlessly; its completion is
        discarded by the generation check in pump(). Synchronous and
        safe from any thread.
        """
        with self._lock:
            self._generation += 1
            self._pending = None
            self._latest = None
            self._held = None
            # Re-anchor the harvest grid to the sim origin. main.reset()
            # zeroes sim_time alongside this call, so the grid restarts
            # identically; a mid-run detector switch re-anchors too, and
            # the closed-form advance snaps back to the absolute grid on
            # the first harvest (no drift either way).
            self._next_harvest_time = self.harvest_interval_s
            self._prev_world = None
            self._last_world = None
            self._noted_fn = None
            self._rate_waz = 0.0
            self._rate_wel = 0.0
            self._track = None
            self._track_fn = None
            self.last_age_ticks = None
            self.last_fresh = False

    async def shutdown(self) -> None:
        """Cancel the worker so no threads/tasks outlive the simulation."""
        with self._lock:
            task, self._task = self._task, None
            self._pending = None
            self._latest = None
            self._held = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # ------------------------------------------------------------------
    # Synchronous detect() — drop-in for simulation_step(detector=...).
    # ------------------------------------------------------------------
    def detect(self, frame) -> Optional[Dict]:
        """Newest-completed-observation decision for the current tick.

        Submits the frame (newest wins) and resolves from completed work
        only; never blocks and never runs inference inline.
        """
        with self._lock:
            self._pending = {
                "frame": frame,
                "sim_time": self._tick_sim_time,
                "frame_number": self._tick_frame,
                "gimbal_az": self._tick_gimbal_az,
                "gimbal_el": self._tick_gimbal_el,
                "generation": self._generation,
            }
            self.frames_submitted += 1
            latest = self._latest
            cur_t = self._tick_sim_time
            cur_fn = self._tick_frame
            cur_gaz = self._tick_gimbal_az
            cur_gel = self._tick_gimbal_el
            max_age_s = self.max_age_s
            generation = self._generation

        if (latest is None or latest["generation"] != generation
                or latest["detection"] is None):
            # Nothing completed yet, stale generation, or a real miss:
            # report undetected (genuine loss path, never masked).
            self.last_age_ticks = None
            self.last_fresh = False
            return None

        # Expiry in simulation seconds (not ticks): the timeout stays
        # invariant when dt changes. last_age_ticks keeps reporting the
        # frame count for telemetry compatibility.
        age_s = cur_t - latest["sim_time"]
        age = cur_fn - latest["frame_number"]
        if age_s < -_TIME_EPS or age_s > max_age_s + _TIME_EPS:
            self.last_age_ticks = None
            self.last_fresh = False
            return None

        det = latest["detection"]
        # Track age explicitly: within the window we coast on the
        # complementary track (fresh only when new information arrived
        # this tick); beyond it the tick is genuinely undetected.
        report, fresh = self._track_resolve(det, latest, cur_t, cur_gaz,
                                            cur_gel)
        if report is None:
            self.last_age_ticks = None
            self.last_fresh = False
            return None
        self.last_age_ticks = age
        self.last_fresh = fresh
        return report

    # ------------------------------------------------------------------
    # World-fixed propagation helpers (mirror Target projection exactly).
    # ------------------------------------------------------------------
    def _to_world(self, bx: float, by: float,
                  gaz: float, gel: float) -> Tuple[float, float]:
        t = self._target
        return ((bx - t.width / 2) / t.pixels_per_rad_x + gaz,
                (t.height / 2 - by) / t.pixels_per_rad_y + gel)

    def _to_pixel(self, waz: float, wel: float,
                  gaz: float, gel: float) -> Optional[Tuple[float, float]]:
        t = self._target
        px = t.width / 2 + (waz - gaz) * t.pixels_per_rad_x
        py = t.height / 2 - (wel - gel) * t.pixels_per_rad_y
        if 0 <= px < t.width and 0 <= py < t.height:
            return (px, py)
        return None

    def _note_world(self, det: Dict, obs: Dict) -> bool:
        """Fold a positive observation into world history, exactly once.

        Returns True when this observation is new (history shifted).
        Also folds the raw two-point world rate into the smoothed rate
        estimate; repeat calls for one observation leave it untouched.
        """
        with self._lock:
            if self._noted_fn == obs["frame_number"]:
                return False
            w = self._to_world(det["x"], det["y"],
                               obs["gimbal_az"], obs["gimbal_el"])
            prev = self._last_world
            if prev is not None and (obs["sim_time"] - prev[2]) > 0:
                dt_obs = obs["sim_time"] - prev[2]
                raw_waz = (w[0] - prev[0]) / dt_obs
                raw_wel = (w[1] - prev[1]) / dt_obs
                if self._prev_world is None:
                    # First measured rate: initialize directly instead of
                    # blending up from zero (avoids systematic
                    # under-prediction on early coasts).
                    self._rate_waz, self._rate_wel = raw_waz, raw_wel
                else:
                    self._rate_waz += self.rate_alpha * (raw_waz - self._rate_waz)
                    self._rate_wel += self.rate_alpha * (raw_wel - self._rate_wel)
            self._prev_world = self._last_world
            self._last_world = (w[0], w[1], obs["sim_time"])
            self._noted_fn = obs["frame_number"]
            return True

    def _track_resolve(self, det: Dict, obs: Dict,
                       cur_t: float, cur_gaz: float, cur_gel: float
                       ) -> Tuple[Optional[Dict], bool]:
        """Complementary world track: predict each tick, correct on news.

        The track is predicted forward with the smoothed measured rate
        every tick; a strictly newer real fix corrects it with
        position_gain and feeds the rate history. The fix is first
        latency-compensated forward to the current tick with the same
        smoothed rate, so the correction is only the residual (rate
        error x age + noise) instead of trailing full observation
        latency behind the target. Spreading arrival corrections over
        ticks keeps the reported stream continuous (no per-tick jumps
        for the controller feedforward to chase) while every correction
        comes from a real observation.

        Returns (report_or_None, fresh) where fresh means new
        information arrived this tick. Out-of-frame predictions report
        None (conservative loss direction).
        """
        with self._lock:
            fix = self._to_world(det["x"], det["y"],
                                 obs["gimbal_az"], obs["gimbal_el"])
            if self._track is None:
                self._track = {"waz": fix[0], "wel": fix[1],
                               "t": obs["sim_time"]}
                self._track_fn = obs["frame_number"]
                self._note_world(det, obs)
                fresh = True
            else:
                dt_pred = cur_t - self._track["t"]
                if dt_pred < 0:
                    return None, False
                self._track["waz"] += self._rate_waz * dt_pred
                self._track["wel"] += self._rate_wel * dt_pred
                self._track["t"] = cur_t
                fresh = False
                if obs["frame_number"] > self._track_fn:
                    fresh = True
                    age_s = cur_t - obs["sim_time"]
                    if age_s < 0:
                        return None, False
                    # Latency compensation: compare against where the
                    # fix should be *now*, not where it was captured.
                    cx = fix[0] + self._rate_waz * age_s
                    cy = fix[1] + self._rate_wel * age_s
                    self._track["waz"] += self.position_gain * (
                        cx - self._track["waz"])
                    self._track["wel"] += self.position_gain * (
                        cy - self._track["wel"])
                    self._track_fn = obs["frame_number"]
                    self._note_world(det, obs)
            pred = self._to_pixel(self._track["waz"], self._track["wel"],
                                  cur_gaz, cur_gel)
            if pred is None:
                return None, False
            return ({"x": pred[0], "y": pred[1],
                     "confidence": det["confidence"], "bbox": None}, fresh)
