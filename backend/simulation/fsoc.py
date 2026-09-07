"""
Free Space Optical Communication (FSOC) link model.
"""
import numpy as np
import logging
from typing import Optional
from dataclasses import dataclass


@dataclass
class LinkQuality:
    """Container for FSO link quality metrics."""
    power_received: float  # Received power [W]
    snr: float  # Signal-to-noise ratio [dB]
    ber: float  # Bit error rate
    link_margin: float  # Link margin [dB]
    is_locked: bool  # Lock status


class FSOCommunication:
    """Models FSO communication link."""
    
    def __init__(self):
        """Initialize FSO communication link."""
        self.logger = logging.getLogger(__name__)
        
        # Transmitter parameters
        self.tx_power = 1.0  # Watts
        self.tx_aperture = 0.1  # meters
        self.tx_divergence = 1e-6  # radians
        
        # Receiver parameters
        self.rx_aperture = 0.1  # meters
        self.rx_efficiency = 0.5
        self.rx_sensitivity = 1e-9  # Watts
        
        # Link parameters
        self.wavelength = 1550e-9  # meters (1550 nm)
        self.range = 1000e3  # meters (1000 km)
        
        # Atmospheric parameters
        self.atmospheric_loss = 0.2  # dB/km
        self.scintillation_index = 0.5
        
        # Pointing error threshold
        self.pointing_threshold = 1e-5  # radians
        
        self.is_locked = False
        
    async def initialize(self):
        """Initialize FSO communication link."""
        self.logger.info("Initializing FSO communication link")
        
    async def update(self,
                    tracking_state,
                    pan_tilt_state) -> LinkQuality:
        """
        Update FSO link quality based on tracking performance.
        
        Args:
            tracking_state: Current tracking state (dict or object)
            pan_tilt_state: Current pan-tilt state (dict or object)
            
        Returns:
            Current link quality metrics
        """
        # Handle both dict and object formats
        if isinstance(tracking_state, dict):
            is_tracking = tracking_state.get('x') is not None
        else:
            is_tracking = getattr(tracking_state, 'tracking', False)
        
        # Calculate pointing error
        pointing_error = self._calculate_pointing_error(
            tracking_state,
            pan_tilt_state
        )
        
        # Check if link is locked
        self.is_locked = (
            is_tracking and 
            pointing_error < self.pointing_threshold
        )
        
        if not self.is_locked:
            return LinkQuality(
                power_received=0.0,
                snr=-np.inf,
                ber=1.0,
                link_margin=-np.inf,
                is_locked=False
            )
        
        # Calculate link budget
        power_received = self._calculate_link_budget(pointing_error)
        
        # Calculate SNR
        snr_db = self._calculate_snr(power_received)
        
        # Calculate BER
        ber = self._calculate_ber(snr_db)
        
        # Calculate link margin
        link_margin = snr_db - self._required_snr()
        
        return LinkQuality(
            power_received=power_received,
            snr=snr_db,
            ber=ber,
            link_margin=link_margin,
            is_locked=True
        )
    
    def _calculate_pointing_error(self,
                                  tracking_state,
                                  pan_tilt_state) -> float:
        """
        Calculate pointing error.
        
        Args:
            tracking_state: Current tracking state
            pan_tilt_state: Current pan-tilt state
            
        Returns:
            Pointing error in radians
        """
        # Simplified pointing error calculation
        # Assume small error for now
        pointing_error = 1e-7  # Very small error
        
        return pointing_error
    
    def _calculate_link_budget(self, pointing_error: float) -> float:
        """
        Calculate received power using link budget.
        
        Args:
            pointing_error: Pointing error [rad]
            
        Returns:
            Received power [W]
        """
        # Geometric spreading loss
        beam_radius = self.tx_divergence * self.range
        geometric_loss = (self.rx_aperture / (2 * beam_radius))**2
        
        # Pointing loss (Gaussian beam approximation)
        pointing_loss = np.exp(-(pointing_error / self.tx_divergence)**2)
        
        # Atmospheric loss
        range_km = self.range / 1000.0
        atmospheric_loss_linear = 10**(-self.atmospheric_loss * range_km / 10)
        
        # Scintillation (log-normal fading)
        scintillation = np.random.lognormal(0, self.scintillation_index)
        
        # Total received power
        power_received = (
            self.tx_power *
            geometric_loss *
            pointing_loss *
            atmospheric_loss_linear *
            scintillation *
            self.rx_efficiency
        )
        
        return power_received
    
    def _calculate_snr(self, power_received: float) -> float:
        """
        Calculate signal-to-noise ratio.
        
        Args:
            power_received: Received power [W]
            
        Returns:
            SNR in dB
        """
        # Noise power (simplified)
        noise_power = 1e-12  # Watts
        
        if power_received <= 0:
            return -np.inf
        
        snr_linear = power_received / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db
    
    def _calculate_ber(self, snr_db: float) -> float:
        """
        Calculate bit error rate.
        
        Args:
            snr_db: SNR in dB
            
        Returns:
            Bit error rate
        """
        if snr_db <= 0:
            return 0.5  # Random guessing
        
        snr_linear = 10**(snr_db / 10)
        
        # BER for OOK (On-Off Keying) modulation
        ber = 0.5 * np.exp(-snr_linear / 2)
        
        return ber
    
    def _required_snr(self) -> float:
        """Get required SNR for target BER."""
        return 15.0  # dB for BER = 1e-9
    
    def set_range(self, range_m: float):
        """Set link range in meters."""
        self.range = range_m
        self.logger.info(f"Link range set to {range_m/1000:.1f} km")
