"""
SQLite database for experiment tracking and history.
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class ExperimentDatabase:
    """Manages experiment data storage and retrieval."""
    
    def __init__(self, db_path: str = "experiments.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration REAL,
                configuration TEXT NOT NULL,
                
                -- Performance metrics
                average_fps REAL,
                average_error REAL,
                rms_error REAL,
                maximum_error REAL,
                minimum_error REAL,
                
                -- Acquisition and tracking
                acquisition_time REAL,
                lock_retention REAL,
                reacquisitions INTEGER,
                
                -- Link quality
                link_availability REAL,
                average_snr REAL,
                average_ber REAL,
                
                -- Disturbance settings
                turbulence REAL,
                vibration REAL,
                camera_motion REAL,
                noise REAL,
                
                -- Detector type
                detector_type TEXT DEFAULT 'classical',
                
                -- Status
                status TEXT DEFAULT 'running',
                notes TEXT
            )
        ''')
        
        # Telemetry data table (for detailed time-series data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS telemetry (
                telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                
                -- Target position
                target_x REAL,
                target_y REAL,
                
                -- Detection
                detected_x REAL,
                detected_y REAL,
                confidence REAL,
                
                -- Tracking
                tracking_error REAL,
                velocity_x REAL,
                velocity_y REAL,
                
                -- Control
                azimuth REAL,
                elevation REAL,
                
                -- Link
                link_status TEXT,
                snr REAL,
                ber REAL,
                
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        ''')
        
        # Create indices for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_experiment_time 
            ON experiments(start_time)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_telemetry_experiment 
            ON telemetry(experiment_id, timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def create_experiment(self, configuration: Dict[str, Any]) -> int:
        """
        Create a new experiment record.
        
        Args:
            configuration: Experiment configuration dict
            
        Returns:
            experiment_id of the new experiment
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO experiments (
                start_time,
                configuration,
                turbulence,
                vibration,
                camera_motion,
                noise,
                detector_type,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps(configuration),
            configuration.get('turbulence', 1.0),
            configuration.get('vibration', 1.0),
            configuration.get('camera_motion', 1.0),
            configuration.get('noise', 1.0),
            configuration.get('detector_type', 'classical'),
            'running'
        ))
        
        experiment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return experiment_id
    
    def update_experiment(self, experiment_id: int, metrics: Dict[str, Any]):
        """
        Update experiment with final metrics.
        
        Args:
            experiment_id: ID of the experiment
            metrics: Dictionary of metrics to update
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        update_fields = []
        values = []
        
        for key, value in metrics.items():
            update_fields.append(f"{key} = ?")
            values.append(value)
        
        values.append(experiment_id)
        
        query = f'''
            UPDATE experiments
            SET {', '.join(update_fields)}
            WHERE experiment_id = ?
        '''
        
        cursor.execute(query, values)
        conn.commit()
        conn.close()
    
    def finish_experiment(
        self,
        experiment_id: int,
        average_fps: float,
        average_error: float,
        rms_error: float,
        maximum_error: float,
        minimum_error: float,
        acquisition_time: float,
        lock_retention: float,
        reacquisitions: int,
        link_availability: float
    ):
        """
        Mark experiment as complete and store final metrics.
        
        Args:
            experiment_id: ID of the experiment
            average_fps: Average frames per second
            average_error: Mean tracking error
            rms_error: Root mean square error
            maximum_error: Maximum tracking error
            minimum_error: Minimum tracking error
            acquisition_time: Time to first lock
            lock_retention: Percentage of time locked
            reacquisitions: Number of re-acquisitions
            link_availability: Percentage of time link was available
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get start time to calculate duration
        cursor.execute(
            'SELECT start_time FROM experiments WHERE experiment_id = ?',
            (experiment_id,)
        )
        start_time_str = cursor.fetchone()[0]
        start_time = datetime.fromisoformat(start_time_str)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        cursor.execute('''
            UPDATE experiments
            SET end_time = ?,
                duration = ?,
                average_fps = ?,
                average_error = ?,
                rms_error = ?,
                maximum_error = ?,
                minimum_error = ?,
                acquisition_time = ?,
                lock_retention = ?,
                reacquisitions = ?,
                link_availability = ?,
                status = 'completed'
            WHERE experiment_id = ?
        ''', (
            end_time.isoformat(),
            duration,
            average_fps,
            average_error,
            rms_error,
            maximum_error,
            minimum_error,
            acquisition_time,
            lock_retention,
            reacquisitions,
            link_availability,
            experiment_id
        ))
        
        conn.commit()
        conn.close()
    
    def add_telemetry(self, experiment_id: int, data: Dict[str, Any]):
        """
        Add telemetry data point.
        
        Args:
            experiment_id: ID of the experiment
            data: Telemetry data dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO telemetry (
                experiment_id, timestamp,
                target_x, target_y,
                detected_x, detected_y, confidence,
                tracking_error, velocity_x, velocity_y,
                azimuth, elevation,
                link_status, snr, ber
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_id,
            data.get('timestamp', 0.0),
            data.get('target_x', 0.0),
            data.get('target_y', 0.0),
            data.get('detected_x', 0.0),
            data.get('detected_y', 0.0),
            data.get('confidence', 0.0),
            data.get('tracking_error', 0.0),
            data.get('velocity_x', 0.0),
            data.get('velocity_y', 0.0),
            data.get('azimuth', 0.0),
            data.get('elevation', 0.0),
            data.get('link_status', 'DISCONNECTED'),
            data.get('snr', 0.0),
            data.get('ber', 1.0)
        ))
        
        conn.commit()
        conn.close()
    
    def get_experiment_list(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get list of recent experiments.
        
        Args:
            limit: Maximum number of experiments to return
            
        Returns:
            List of experiment dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                experiment_id,
                start_time,
                duration,
                lock_retention,
                average_error,
                turbulence,
                vibration,
                camera_motion,
                noise,
                detector_type,
                status
            FROM experiments
            ORDER BY start_time DESC
            LIMIT ?
        ''', (limit,))
        
        experiments = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return experiments
    
    def get_experiment_details(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Experiment details dictionary or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM experiments WHERE experiment_id = ?',
            (experiment_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_telemetry_data(
        self,
        experiment_id: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get telemetry data for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            
        Returns:
            List of telemetry data points
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = 'SELECT * FROM telemetry WHERE experiment_id = ?'
        params = [experiment_id]
        
        if start_time is not None:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        
        if end_time is not None:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        
        query += ' ORDER BY timestamp ASC'
        
        cursor.execute(query, params)
        telemetry = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return telemetry
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics across all experiments.
        
        Returns:
            Dictionary with summary statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_experiments,
                AVG(lock_retention) as avg_lock_retention,
                AVG(average_error) as avg_error,
                AVG(average_fps) as avg_fps,
                MAX(lock_retention) as best_lock_retention,
                MIN(average_error) as best_error
            FROM experiments
            WHERE status = 'completed'
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            'total_experiments': row[0] or 0,
            'avg_lock_retention': row[1] or 0.0,
            'avg_error': row[2] or 0.0,
            'avg_fps': row[3] or 0.0,
            'best_lock_retention': row[4] or 0.0,
            'best_error': row[5] or 999.9
        }
