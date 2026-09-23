"""
Pydantic schemas for configuration and data models.

NOTE: These schemas are currently informational only (nothing imports
them at runtime). Defaults mirror the live simulation
(SimulationManager at 30 Hz, VirtualCamera 1280x720, Target FOV
40x30 deg, PATController acquire-mode gains) so they stay truthful.
"""
from typing import Optional, Tuple
from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    """Configuration for the simulation."""

    # Time parameters
    time_step: float = Field(default=1.0 / 30.0, description="Simulation time step [s]")
    duration: Optional[float] = Field(default=None, description="Simulation duration [s]")

    # Target parameters
    target_orbital_period: float = Field(default=6000.0, description="Target orbital period [s]")
    target_semi_major_axis: float = Field(default=7000e3, description="Semi-major axis [m]")

    # Camera parameters (mirror VirtualCamera + Target live values)
    camera_resolution: Tuple[int, int] = Field(default=(1280, 720), description="Camera resolution [w, h] px")
    camera_frame_rate: float = Field(default=30.0, description="Camera frame rate [Hz]")
    camera_fov_horizontal_deg: float = Field(default=40.0, description="Horizontal field of view [deg]")
    camera_fov_vertical_deg: float = Field(default=30.0, description="Vertical field of view [deg]")

    # Detector parameters
    detection_threshold: float = Field(default=5.0, description="Detection SNR threshold")

    # PID gains (mirror PATController acquire-mode gains)
    pid_kp: float = Field(default=0.10, description="Proportional gain")
    pid_ki: float = Field(default=0.002, description="Integral gain")
    pid_kd: float = Field(default=0.02, description="Derivative gain")
    
    # FSO parameters
    fso_wavelength: float = Field(default=1550e-9, description="Laser wavelength [m]")
    fso_tx_power: float = Field(default=1.0, description="Transmit power [W]")
    fso_range: float = Field(default=1000e3, description="Link range [m]")
    
    # Disturbance parameters
    jitter_amplitude: float = Field(default=1e-6, description="Jitter amplitude [rad]")
    vibration_amplitude: float = Field(default=1e-4, description="Vibration amplitude [m/s^2]")
    atmospheric_strength: float = Field(default=0.5, description="Atmospheric turbulence strength")
    
    # Output parameters
    telemetry_output_dir: str = Field(default="telemetry", description="Telemetry output directory")
    log_level: str = Field(default="INFO", description="Logging level")


class TargetState(BaseModel):
    """Target state representation."""
    position: Tuple[float, float, float] = Field(description="Position [x, y, z] in meters")
    velocity: Tuple[float, float, float] = Field(description="Velocity [vx, vy, vz] in m/s")
    time: float = Field(description="Time [s]")


class TrackingData(BaseModel):
    """Tracking data representation."""
    position: Tuple[float, float] = Field(description="Pixel position [x, y]")
    velocity: Tuple[float, float] = Field(description="Pixel velocity [vx, vy]")
    tracking: bool = Field(description="Tracking status")
    confidence: float = Field(ge=0.0, le=1.0, description="Tracking confidence")


class LinkStatus(BaseModel):
    """FSO link status."""
    power_received: float = Field(description="Received power [W]")
    snr: float = Field(description="Signal-to-noise ratio [dB]")
    ber: float = Field(ge=0.0, le=1.0, description="Bit error rate")
    link_margin: float = Field(description="Link margin [dB]")
    is_locked: bool = Field(description="Link lock status")


class TelemetryRecord(BaseModel):
    """Single telemetry record."""
    time: float = Field(description="Simulation time [s]")
    target: TargetState
    tracking: TrackingData
    link: LinkStatus
