"""Deterministic unit tests for simulation/link_budget.py.

No randomness, no I/O, no PAT/controller/detector involvement: every test
calls link_budget() twice where equality matters and asserts physically
necessary relationships (monotonicity, exact dB identities, scaling laws).
"""
import json
import math

import pytest

from simulation.link_budget import LinkBudgetConfig, SNR_FLOOR_DB, link_budget


def test_perfect_alignment():
    """Zero pointing error means zero pointing loss and a NOMINAL link."""
    result = link_budget(0.0)
    assert result["pointing_loss_db"] == pytest.approx(0.0, abs=1e-9)
    assert result["received_power_w"] > 0.0
    assert result["link_state"] == "NOMINAL"
    # Atmospheric identity holds exactly: loss_dB == rate * path.
    assert result["atmospheric_loss_db"] == pytest.approx(0.5 * 10.0)
    # Pointing error of exactly one divergence = 10*log10(e) dB by definition.
    cfg = LinkBudgetConfig()
    one_div = link_budget(cfg.divergence_rad, cfg)
    assert one_div["pointing_loss_db"] == pytest.approx(10.0 * math.log10(math.e))


def test_increasing_pointing_error():
    """Larger mispointing strictly reduces power and degrades the state."""
    cfg = LinkBudgetConfig()
    div = cfg.divergence_rad
    errors = [0.0, 0.25 * div, 0.5 * div, div, 2.0 * div, 1e-3]
    powers = [link_budget(e, cfg)["received_power_w"] for e in errors]
    assert all(a > b for a, b in zip(powers, powers[1:]))
    losses = [link_budget(e, cfg)["pointing_loss_db"] for e in errors]
    assert all(a < b for a, b in zip(losses, losses[1:]))
    states = [link_budget(e, cfg)["link_state"] for e in errors]
    assert states[0] == "NOMINAL"
    assert states[-1] == "OUTAGE"
    # State order never regresses NOMINAL <- MARGINAL <- OUTAGE.
    rank = {"NOMINAL": 0, "MARGINAL": 1, "OUTAGE": 2}
    assert [rank[s] for s in states] == sorted(rank[s] for s in states)


def test_increasing_range():
    """Inverse-square law: doubling range quarters power, +6.02 dB geometry."""
    cfg = LinkBudgetConfig()
    near = link_budget(0.0, cfg)
    import dataclasses

    far_cfg = dataclasses.replace(cfg, range_m=2.0 * cfg.range_m)
    far = link_budget(0.0, far_cfg)
    assert far["received_power_w"] == pytest.approx(near["received_power_w"] / 4.0)
    assert far["geometric_loss_db"] == pytest.approx(
        near["geometric_loss_db"] + 20.0 * math.log10(2.0)
    )
    # Transmit-power linearity: twice the power in, twice out.
    double_tx = link_budget(
        0.0, dataclasses.replace(cfg, tx_power_w=2.0 * cfg.tx_power_w)
    )
    assert double_tx["received_power_w"] == pytest.approx(2.0 * near["received_power_w"])
    # Near-field clamp: a passive link can never deliver more than sent.
    close = link_budget(0.0, dataclasses.replace(cfg, range_m=1.0))
    assert close["received_power_w"] <= cfg.tx_power_w


def test_atmospheric_loss():
    """Atmospheric dB scales exactly with rate and path; power follows."""
    base = LinkBudgetConfig(atm_attenuation_db_per_km=0.0)
    assert link_budget(0.0, base)["atmospheric_loss_db"] == pytest.approx(0.0)
    import dataclasses

    foggy = dataclasses.replace(base, atm_attenuation_db_per_km=2.0, atm_path_km=5.0)
    foggy_result = link_budget(0.0, foggy)
    assert foggy_result["atmospheric_loss_db"] == pytest.approx(10.0)
    clear_result = link_budget(0.0, base)
    assert foggy_result["received_power_w"] == pytest.approx(
        clear_result["received_power_w"] * 10.0 ** (-10.0 / 10.0)
    )
    assert foggy_result["received_power_w"] < clear_result["received_power_w"]


def test_degraded_and_failed_link():
    """Margin bands and failure are set by SNR thresholds, not PAT state."""
    import dataclasses

    cfg = LinkBudgetConfig()
    nominal = link_budget(0.0, cfg)
    assert nominal["link_state"] == "NOMINAL"
    # Back the required SNR off to just below measured SNR -> MARGINAL band.
    marginal = link_budget(
        0.0, dataclasses.replace(cfg, required_snr_db=nominal["snr_db"] - 1.0)
    )
    assert marginal["link_state"] == "MARGINAL"
    assert marginal["link_margin_db"] == pytest.approx(1.0)
    # Total mispointing kills the signal: floor SNR, negative margin, OUTAGE.
    failed = link_budget(1e-2, cfg)
    assert failed["received_power_w"] == pytest.approx(0.0, abs=1e-300)
    assert failed["snr_db"] == SNR_FLOOR_DB
    assert failed["link_margin_db"] < 0.0
    assert failed["link_state"] == "OUTAGE"
    # Deep fade input attenuates linearly and deterministically.
    half_fade = link_budget(
        0.0, dataclasses.replace(cfg, scintillation_factor=0.5)
    )
    assert half_fade["received_power_w"] == pytest.approx(
        0.5 * nominal["received_power_w"]
    )


def test_deterministic_json_and_validation():
    """Same inputs give identical, JSON-serializable outputs; bad inputs raise."""
    cfg = LinkBudgetConfig()
    assert link_budget(1e-6, cfg) == link_budget(1e-6, cfg)
    json.dumps(link_budget(1e-6, cfg))  # must not raise
    with pytest.raises(ValueError):
        link_budget(-1e-6, cfg)
    with pytest.raises(ValueError):
        link_budget(float("nan"), cfg)
    with pytest.raises(ValueError):
        LinkBudgetConfig(tx_power_w=0.0)
    with pytest.raises(ValueError):
        LinkBudgetConfig(range_m=-5.0)
    with pytest.raises(ValueError):
        LinkBudgetConfig(scintillation_factor=0.0)
    with pytest.raises(ValueError):
        LinkBudgetConfig(scintillation_factor=1.5)
    with pytest.raises(ValueError):
        LinkBudgetConfig(optical_efficiency=1.5)
