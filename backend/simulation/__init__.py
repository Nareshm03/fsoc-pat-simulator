"""
Simulation package for satellite tracking and FSO communication.
"""
from .target import Target
from .camera import VirtualCamera
from .detector import BeaconDetector
from .yolo_detector import YOLO11Detector, create_detector, AVAILABLE_DETECTORS
from .tracker import Tracker
from .pat_controller import PATController
from .pan_tilt import PanTilt
from .sim_step import simulation_step
from .async_detector import (
    AsyncDetector,
    MAX_OBSERVATION_AGE_S,
    MAX_OBSERVATION_AGE_TICKS,
)

__all__ = [
    "Target",
    "VirtualCamera",
    "BeaconDetector",
    "YOLO11Detector",
    "create_detector",
    "AVAILABLE_DETECTORS",
    "Tracker",
    "PATController",
    "PanTilt",
    "simulation_step",
    "AsyncDetector",
    "MAX_OBSERVATION_AGE_S",
    "MAX_OBSERVATION_AGE_TICKS",
]
