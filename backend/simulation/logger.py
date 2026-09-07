"""
Logging configuration for the simulation.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(level: int = logging.INFO, log_file: bool = True):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Whether to log to file
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = log_dir / f"simulation_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {log_filename}")
    
    return root_logger
