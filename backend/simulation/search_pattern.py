"""Deterministic bounded raster search for lost-beacon recovery.

When the PAT state machine reports SEARCH/REACQUIRE (beacon genuinely
undetected), the gimbal sweeps a serpentine raster instead of holding
still. Pure function of (gimbal position, simulation time): no randomness,
no detector input, no fabricated detections. All targets lie inside the
configured bounds, which themselves lie inside the PanTilt position
limits; velocity commands are rate-capped at or below the PAT and gimbal
rate limits.
"""

from __future__ import annotations

import math


class RasterSearch:
    """Serpentine azimuth sweep with stepped elevation rows (ping-pong)."""

    def __init__(
        self,
        az_min_deg: float = -20.0,
        az_max_deg: float = 20.0,
        el_min_deg: float = -10.0,
        el_max_deg: float = 10.0,
        az_sweep_rate_deg_s: float = 6.0,
        el_step_deg: float = 2.0,
        max_rate_deg_s: float = 10.0,
    ) -> None:
        if not math.isfinite(az_min_deg) or not math.isfinite(az_max_deg):
            raise ValueError("az bounds must be finite")
        if not math.isfinite(el_min_deg) or not math.isfinite(el_max_deg):
            raise ValueError("el bounds must be finite")
        if az_min_deg >= az_max_deg:
            raise ValueError("az_min_deg must be < az_max_deg")
        if el_min_deg > el_max_deg:
            raise ValueError("el_min_deg must be <= el_max_deg")
        for name, value in (
            ("az_sweep_rate_deg_s", az_sweep_rate_deg_s),
            ("el_step_deg", el_step_deg),
            ("max_rate_deg_s", max_rate_deg_s),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self.az_min = float(az_min_deg)
        self.az_max = float(az_max_deg)
        self.el_min = float(el_min_deg)
        self.el_max = float(el_max_deg)
        self.az_sweep_rate = float(az_sweep_rate_deg_s)
        self.max_rate = float(max_rate_deg_s)
        self.n_rows = max(1, int(round((self.el_max - self.el_min) / el_step_deg)))
        self.row_step = (
            (self.el_max - self.el_min) / (self.n_rows - 1) if self.n_rows > 1 else 0.0
        )
        self.row_period = (self.az_max - self.az_min) / self.az_sweep_rate

    def target_at(self, t: float) -> tuple:
        """Serpentine waypoint (az_deg, el_deg) at simulation time t."""
        if not math.isfinite(t):
            raise ValueError(f"t must be finite, got {t!r}")
        t = max(0.0, t)
        cycle = 2 * self.n_rows
        row = int(math.floor(t / self.row_period)) % cycle
        eff_row = row if row < self.n_rows else cycle - 1 - row
        frac = (t % self.row_period) / self.row_period
        width = self.az_max - self.az_min
        az = self.az_min + frac * width if eff_row % 2 == 0 else self.az_max - frac * width
        el = min(self.el_max, self.el_min + eff_row * self.row_step)
        return (az, el)

    def command(
        self, gimbal_az_deg: float, gimbal_el_deg: float, t: float, dt: float
    ) -> tuple:
        """Rate-capped velocity command driving the gimbal at the waypoint."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"dt must be positive and finite, got {dt!r}")
        target_az, target_el = self.target_at(t)

        def _axis_rate(diff: float) -> float:
            cmd = diff / dt
            return max(-self.max_rate, min(self.max_rate, cmd))

        return (
            _axis_rate(target_az - gimbal_az_deg),
            _axis_rate(target_el - gimbal_el_deg),
        )
