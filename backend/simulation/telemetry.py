"""
Telemetry collection and logging.
"""
import json
import logging
from typing import Dict, List, Any
from pathlib import Path
import numpy as np


class TelemetryCollector:
    """Collects and saves simulation telemetry data."""
    
    def __init__(self, output_dir: str = "telemetry"):
        """
        Initialize telemetry collector.
        
        Args:
            output_dir: Directory to save telemetry files
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data: List[Dict[str, Any]] = []
        
    async def record(self,
                    time: float,
                    target_position,
                    tracking_state,
                    control_signal,
                    link_quality):
        """
        Record telemetry data point.
        
        Args:
            time: Simulation time
            target_position: Target position vector or tuple
            tracking_state: Current tracking state (dict or object)
            control_signal: Control signal (dict or object)
            link_quality: FSO link quality (dict or object)
        """
        # Handle different input types
        if isinstance(target_position, (list, tuple)):
            target_x, target_y = target_position[0], target_position[1]
            target_z = target_position[2] if len(target_position) > 2 else 0.0
        else:
            target_x, target_y, target_z = target_position[0], target_position[1], target_position[2] if len(target_position) > 2 else 0.0
            
        # Handle tracking state
        if isinstance(tracking_state, dict):
            track_x = tracking_state.get('x', 0.0)
            track_y = tracking_state.get('y', 0.0)
            track_vx = tracking_state.get('vx', 0.0)
            track_vy = tracking_state.get('vy', 0.0)
            is_tracking = tracking_state.get('x') is not None
        else:
            track_x = tracking_state.position[0] if hasattr(tracking_state, 'position') else 0.0
            track_y = tracking_state.position[1] if hasattr(tracking_state, 'position') else 0.0
            track_vx = tracking_state.velocity[0] if hasattr(tracking_state, 'velocity') else 0.0
            track_vy = tracking_state.velocity[1] if hasattr(tracking_state, 'velocity') else 0.0
            is_tracking = tracking_state.tracking if hasattr(tracking_state, 'tracking') else False
            
        # Handle control signal
        if isinstance(control_signal, dict):
            azimuth = control_signal.get('azimuth', 0.0)
            elevation = control_signal.get('elevation', 0.0)
            mode = control_signal.get('mode', 'idle')
        else:
            azimuth = control_signal.azimuth if hasattr(control_signal, 'azimuth') else 0.0
            elevation = control_signal.elevation if hasattr(control_signal, 'elevation') else 0.0
            mode = control_signal.mode if hasattr(control_signal, 'mode') else 'idle'
            
        # Handle link quality
        if isinstance(link_quality, dict):
            power = link_quality.get('power_received', 0.0)
            snr = link_quality.get('snr', -np.inf)
            ber = link_quality.get('ber', 1.0)
            margin = link_quality.get('link_margin', -np.inf)
            locked = link_quality.get('is_locked', False)
        else:
            power = link_quality.power_received if hasattr(link_quality, 'power_received') else 0.0
            snr = link_quality.snr if hasattr(link_quality, 'snr') else -np.inf
            ber = link_quality.ber if hasattr(link_quality, 'ber') else 1.0
            margin = link_quality.link_margin if hasattr(link_quality, 'link_margin') else -np.inf
            locked = link_quality.is_locked if hasattr(link_quality, 'is_locked') else False
        
        record = {
            'time': time,
            'target': {
                'x': float(target_x),
                'y': float(target_y),
                'z': float(target_z),
            },
            'tracking': {
                'position_x': float(track_x),
                'position_y': float(track_y),
                'velocity_x': float(track_vx),
                'velocity_y': float(track_vy),
                'tracking': is_tracking,
            },
            'control': {
                'azimuth': float(azimuth),
                'elevation': float(elevation),
                'mode': mode,
            },
            'link': {
                'power_received': float(power),
                'snr': float(snr) if np.isfinite(snr) else None,
                'ber': float(ber),
                'link_margin': float(margin) if np.isfinite(margin) else None,
                'is_locked': locked,
            }
        }
        
        self.data.append(record)
    
    async def save(self, filename: str = "telemetry.json"):
        """
        Save telemetry data to file.
        
        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            
            self.logger.info(f"Telemetry saved to {output_path}")
            self.logger.info(f"Total records: {len(self.data)}")
            
        except Exception as e:
            self.logger.error(f"Failed to save telemetry: {e}")
    
    def clear(self):
        """Clear collected telemetry data."""
        self.data.clear()
        self.logger.info("Telemetry data cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of telemetry data.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.data:
            return {}
        
        # Extract metrics
        link_locked_count = sum(1 for d in self.data if d['link']['is_locked'])
        tracking_count = sum(1 for d in self.data if d['tracking']['tracking'])
        
        snr_values = [d['link']['snr'] for d in self.data if d['link']['snr'] is not None]
        
        summary = {
            'total_records': len(self.data),
            'duration': self.data[-1]['time'] if self.data else 0.0,
            'link_locked_percent': 100 * link_locked_count / len(self.data),
            'tracking_percent': 100 * tracking_count / len(self.data),
            'average_snr': np.mean(snr_values) if snr_values else None,
            'min_snr': np.min(snr_values) if snr_values else None,
            'max_snr': np.max(snr_values) if snr_values else None,
        }
        
        return summary
