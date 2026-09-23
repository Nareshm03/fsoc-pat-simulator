"""Deterministic free-space optical (FSO) link-budget model.

Pure simulation module: standard library only, no randomness, no I/O,
no dependency on PAT/controller/detector state. The link state is derived
solely from the computed link margin — never from any acquisition or lock
flag passed in by a caller.

============================ CONFIGURABLE ASSUMPTIONS ============================
A1. Aperture gains follow the ideal circular-aperture Friis form
    G = (pi * D / wavelength)^2 for both ends.
A2. Default beam divergence is the diffraction limit 1.22 * lambda / D_tx.
    An explicit divergence may be configured instead.
A3. Atmospheric loss applies over a configurable slant path through the
    atmosphere (default 10 km), NOT over the full link range. A 1000 km
    space link only traverses air near the ground terminal.
A4. Pointing loss assumes a Gaussian beam profile:
    L_point = exp(-(pointing_error / divergence)^2).
A5. Scintillation/fading is a caller-supplied deterministic factor in
    (0, 1] (1.0 = no fading). This model generates no randomness; a
    time-varying fade must be computed by the caller.
A6. Receiver noise is a single configurable power (thermal + background,
    default 1e-12 W). SNR = received_power / noise_power.
A7. link_margin = snr_db - required_snr_db. The required SNR is an
    operator threshold, NOT derived from any bit-error claim.
A8. link_state bands are pure margin thresholds (configurable):
    margin >= nominal_margin_db -> "NOMINAL",
    margin >= 0                -> "MARGINAL",
    otherwise                  -> "OUTAGE".

================================= NON-GOALS ====================================
- No BER model: BER requires explicit modulation, coding, and noise-spectrum
  assumptions, which are deliberately out of scope. Use snr_db / link_margin.
- No PAT coupling: pointing_error_rad is a plain float input.
- Numerical guard only: SNR dB is floored at SNR_FLOOR_DB when the received
  power is zero (log of zero is undefined); this is not physics.

================================= EQUATIONS ====================================
E1. G_tx  = (pi * tx_aperture_m / wavelength_m)^2
E2. G_rx  = (pi * rx_aperture_m / wavelength_m)^2
E3. L_fs  = (wavelength_m / (4 * pi * range_m))^2        (free-space loss)
E4. g_path = min(1, G_tx * G_rx * L_fs)                  (near-field clamp)
E5. L_atm  = 10^(-atm_attenuation_db_per_km * atm_path_km / 10)
E6. L_point = exp(-(pointing_error_rad / divergence)^2)
E7. P_rx = tx_power_w * g_path * L_atm * L_point
          * optical_efficiency * scintillation_factor      (capped at tx power)
E8. snr_db = 10*log10(P_rx / noise_power_w), floored at SNR_FLOOR_DB
E9. link_margin_db = snr_db - required_snr_db
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

#: Numerical floor for SNR dB when received power is zero. Guard only.
SNR_FLOOR_DB = -120.0


def _positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return float(value)


def _unit_interval(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")
    return float(value)


@dataclass(frozen=True)
class LinkBudgetConfig:
    """All model inputs. Defaults describe a 1000 km, 1550 nm terminal pair."""

    tx_power_w: float = 1.0
    wavelength_m: float = 1550e-9
    range_m: float = 1000e3
    tx_aperture_m: float = 0.1
    rx_aperture_m: float = 0.1
    optical_efficiency: float = 0.5
    atm_attenuation_db_per_km: float = 0.5
    atm_path_km: float = 10.0
    beam_divergence_rad: Optional[float] = None  # None -> diffraction limit (A2)
    scintillation_factor: float = 1.0  # deterministic fade input, see A5
    receiver_noise_power_w: float = 1e-12  # see A6
    required_snr_db: float = 10.0  # operator threshold, see A7
    nominal_margin_db: float = 3.0  # NOMINAL band edge, see A8

    def __post_init__(self) -> None:
        _positive("tx_power_w", self.tx_power_w)
        _positive("wavelength_m", self.wavelength_m)
        _positive("range_m", self.range_m)
        _positive("tx_aperture_m", self.tx_aperture_m)
        _positive("rx_aperture_m", self.rx_aperture_m)
        _unit_interval("optical_efficiency", self.optical_efficiency)
        if not math.isfinite(self.atm_attenuation_db_per_km) or self.atm_attenuation_db_per_km < 0:
            raise ValueError("atm_attenuation_db_per_km must be finite and >= 0")
        if not math.isfinite(self.atm_path_km) or self.atm_path_km < 0:
            raise ValueError("atm_path_km must be finite and >= 0")
        if self.beam_divergence_rad is not None:
            _positive("beam_divergence_rad", self.beam_divergence_rad)
        _unit_interval("scintillation_factor", self.scintillation_factor)
        _positive("receiver_noise_power_w", self.receiver_noise_power_w)
        for name in ("required_snr_db", "nominal_margin_db"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")

    @property
    def divergence_rad(self) -> float:
        """Effective 1/e beam divergence (A2)."""
        if self.beam_divergence_rad is not None:
            return self.beam_divergence_rad
        return 1.22 * self.wavelength_m / self.tx_aperture_m


def link_budget(
    pointing_error_rad: float,
    config: Optional[LinkBudgetConfig] = None,
) -> Dict[str, Any]:
    """Evaluate the link for one angular pointing error (radians).

    Returns a JSON-serializable dict with received_power_w, pointing_loss_db,
    atmospheric_loss_db, link_margin_db, snr_db (+snr_linear), link_state,
    plus geometric_loss_db and the applied scintillation factor.
    Losses are reported as positive dB magnitudes.
    """
    cfg = config if config is not None else LinkBudgetConfig()
    if not math.isfinite(pointing_error_rad) or pointing_error_rad < 0:
        raise ValueError(f"pointing_error_rad must be finite and >= 0, got {pointing_error_rad!r}")

    # E1-E3: aperture gains and free-space loss.
    g_tx = (math.pi * cfg.tx_aperture_m / cfg.wavelength_m) ** 2
    g_rx = (math.pi * cfg.rx_aperture_m / cfg.wavelength_m) ** 2
    l_fs = (cfg.wavelength_m / (4.0 * math.pi * cfg.range_m)) ** 2

    # E4: path gain with near-field clamp (a passive link cannot amplify).
    path_gain = min(1.0, g_tx * g_rx * l_fs)
    geometric_loss_db = -10.0 * math.log10(path_gain) if path_gain > 0 else math.inf

    # E5: atmospheric loss over the slant path only (A3).
    atmospheric_loss_db = cfg.atm_attenuation_db_per_km * cfg.atm_path_km
    l_atm = 10.0 ** (-atmospheric_loss_db / 10.0)

    # E6: Gaussian mispointing loss (A4).
    divergence = cfg.divergence_rad
    l_point = math.exp(-((pointing_error_rad / divergence) ** 2))
    pointing_loss_db = -10.0 * math.log10(l_point) if l_point > 0 else math.inf

    # E7: received power, capped at transmitted power (passivity).
    received_power_w = min(
        cfg.tx_power_w,
        cfg.tx_power_w * path_gain * l_atm * l_point
        * cfg.optical_efficiency * cfg.scintillation_factor,
    )

    # E8-E9: SNR, margin, and margin-derived state (A6-A8).
    snr_linear = received_power_w / cfg.receiver_noise_power_w
    snr_db = 10.0 * math.log10(snr_linear) if snr_linear > 0 else SNR_FLOOR_DB
    snr_db = max(snr_db, SNR_FLOOR_DB)
    link_margin_db = snr_db - cfg.required_snr_db
    if link_margin_db >= cfg.nominal_margin_db:
        link_state = "NOMINAL"
    elif link_margin_db >= 0.0:
        link_state = "MARGINAL"
    else:
        link_state = "OUTAGE"

    return {
        "received_power_w": received_power_w,
        "pointing_loss_db": pointing_loss_db,
        "atmospheric_loss_db": atmospheric_loss_db,
        "geometric_loss_db": geometric_loss_db,
        "link_margin_db": link_margin_db,
        "snr_db": snr_db,
        "snr_linear": snr_linear,
        "link_state": link_state,
        "scintillation_factor": cfg.scintillation_factor,
        "beam_divergence_rad": divergence,
    }
