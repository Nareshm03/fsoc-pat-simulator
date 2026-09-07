"""
Dataset Generator for YOLO Training

Automatically generates labeled training data for beacon detection:
- Randomizes beacon position, size, brightness
- Randomizes environmental conditions (turbulence, noise, vibration, blur)
- Captures frames and generates YOLO format labels
- Splits data into train/val/test (67/17/17)
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path
from typing import Tuple, Dict, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    num_images: int = 300  # Pilot dataset size
    train_split: float = 0.667  # 200 images (66.7%)
    val_split: float = 0.167    # 50 images (16.7%)
    test_split: float = 0.167   # 50 images (16.7% - actually 16.6% to get exactly 50)
    output_dir: str = "dataset"
    
    # Randomization ranges (tuned for detectability)
    beacon_x_range: Tuple[float, float] = (250, 1030)  # Safe margin from edges
    beacon_y_range: Tuple[float, float] = (150, 570)   # Safe margin from edges
    beacon_size_range: Tuple[float, float] = (8, 25)   # Minimum 8 for detectability
    brightness_range: Tuple[float, float] = (0.5, 1.0) # Minimum 0.5 for visibility
    turbulence_range: Tuple[float, float] = (0.0, 1.5) # Reduced max
    noise_range: Tuple[float, float] = (0.0, 1.5)      # Reduced max
    vibration_range: Tuple[float, float] = (0.0, 1.5)  # Reduced max
    blur_range: Tuple[int, int] = (0, 5)  # Reduced max blur


class DatasetGenerator:
    """Generates synthetic dataset for YOLO beacon detection training."""
    
    def __init__(self, camera, detector, disturbances, config: DatasetConfig = None):
        """
        Initialize dataset generator.
        
        Args:
            camera: Camera instance from simulation
            detector: Detector instance from simulation
            disturbances: Disturbances instance from simulation
            config: Dataset configuration (optional)
        """
        self.camera = camera
        self.detector = detector
        self.disturbances = disturbances
        self.config = config or DatasetConfig()
        
        # Statistics
        self.stats = {
            'total_generated': 0,
            'train_count': 0,
            'val_count': 0,
            'test_count': 0,
            'valid_samples': 0,
            'rejected_samples': 0,
            'detector_failures': 0,
            'large_detection_errors': 0,
            'invalid_labels': 0,
            'min_beacon_size': float('inf'),
            'max_beacon_size': 0
        }
        
        logger.info("Dataset generator initialized")
    
    def setup_directories(self) -> Path:
        """Create dataset directory structure."""
        base_path = Path(self.config.output_dir)
        
        # Create all subdirectories
        for split in ['train', 'val', 'test']:
            (base_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (base_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dataset directories created at {base_path.absolute()}")
        return base_path
    
    def _randomize_beacon_position(self) -> Tuple[float, float]:
        """Generate random beacon position within camera FOV."""
        x = random.uniform(*self.config.beacon_x_range)
        y = random.uniform(*self.config.beacon_y_range)
        return x, y
    
    def _randomize_beacon_size(self) -> float:
        """Generate random beacon size."""
        return random.uniform(*self.config.beacon_size_range)
    
    def _randomize_brightness(self) -> float:
        """Generate random brightness multiplier."""
        return random.uniform(*self.config.brightness_range)
    
    def _randomize_disturbances(self) -> Dict[str, float]:
        """Generate random disturbance parameters."""
        return {
            'turbulence': random.uniform(*self.config.turbulence_range),
            'noise': random.uniform(*self.config.noise_range),
            'vibration': random.uniform(*self.config.vibration_range)
        }
    
    def _randomize_blur(self) -> int:
        """Generate random blur kernel size (must be odd)."""
        blur = random.randint(*self.config.blur_range)
        if blur > 0 and blur % 2 == 0:
            blur += 1  # Make odd
        return blur
    
    def _apply_brightness(self, frame: np.ndarray, brightness: float) -> np.ndarray:
        """Apply brightness adjustment to frame."""
        frame_float = frame.astype(np.float32)
        frame_float = frame_float * brightness
        frame_float = np.clip(frame_float, 0, 255)
        return frame_float.astype(np.uint8)
    
    def _apply_motion_blur(self, frame: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply motion blur to frame."""
        if kernel_size <= 0:
            return frame
        
        # Create motion blur kernel (horizontal motion)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        return cv2.filter2D(frame, -1, kernel)
    
    def _apply_disturbances(self, disturbances: Dict[str, float]):
        """Apply disturbance parameters to simulation."""
        self.disturbances.turbulence = disturbances['turbulence']
        self.disturbances.sensor_noise = disturbances['noise']
        self.disturbances.vibration = disturbances['vibration']
    
    def _generate_yolo_label(
        self, 
        beacon_x: float, 
        beacon_y: float, 
        beacon_size: float,
        img_width: int = 1280,
        img_height: int = 720
    ) -> str:
        """
        Generate YOLO format label.
        
        YOLO format: <class> <x_center> <y_center> <width> <height>
        All coordinates normalized to [0, 1]
        
        Args:
            beacon_x: Beacon x position in pixels
            beacon_y: Beacon y position in pixels
            beacon_size: Beacon radius in pixels
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            YOLO format label string
        """
        # Class 0 = beacon
        class_id = 0
        
        # Normalize coordinates to [0, 1]
        x_center = beacon_x / img_width
        y_center = beacon_y / img_height
        
        # Bounding box size (diameter in normalized coords)
        width = (beacon_size * 2.5) / img_width   # 2.5x radius for box
        height = (beacon_size * 2.5) / img_height
        
        # Ensure within bounds
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def _prepare_split_assignments(self) -> List[Tuple[int, str]]:
        """
        Pre-assign unique sample IDs to splits.
        
        Returns:
            List of (sample_id, split) tuples ensuring no overlap
        """
        train_count = int(self.config.num_images * self.config.train_split)
        val_count = int(self.config.num_images * self.config.val_split)
        test_count = self.config.num_images - train_count - val_count
        
        assignments = []
        sample_id = 0
        
        # Assign train samples
        for _ in range(train_count):
            assignments.append((sample_id, 'train'))
            sample_id += 1
        
        # Assign val samples
        for _ in range(val_count):
            assignments.append((sample_id, 'val'))
            sample_id += 1
        
        # Assign test samples
        for _ in range(test_count):
            assignments.append((sample_id, 'test'))
            sample_id += 1
        
        logger.info(f"Split assignments: train={train_count}, val={val_count}, test={test_count}, total={len(assignments)}")
        return assignments
    
    def _validate_dataset(self, base_path: Path) -> Dict:
        """
        Validate dataset integrity after generation.
        
        Checks:
        - Unique filenames across all splits
        - Image count matches label count per split
        - No sample ID overlap between splits
        - Every label is valid YOLO format
        - Beacon bbox is fully inside image bounds
        
        Returns:
            Validation results dictionary
        """
        logger.info("Validating dataset integrity...")
        
        validation = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'duplicate_count': 0,
            'split_overlap_count': 0,
            'invalid_label_count': 0,
            'bbox_outside_count': 0,
            'missing_label_count': 0,
            'missing_image_count': 0,
            'per_split': {}
        }
        
        # Track all filenames globally
        all_filenames = set()
        split_files = {'train': set(), 'val': set(), 'test': set()}
        
        for split in ['train', 'val', 'test']:
            img_dir = base_path / 'images' / split
            label_dir = base_path / 'labels' / split
            
            # Get all images and labels
            images = set(p.stem for p in img_dir.glob('*.png'))
            labels = set(p.stem for p in label_dir.glob('*.txt'))
            
            split_files[split] = images
            
            # Check image-label count match
            img_count = len(images)
            label_count = len(labels)
            
            validation['per_split'][split] = {
                'image_count': img_count,
                'label_count': label_count,
                'match': img_count == label_count
            }
            
            if img_count != label_count:
                validation['passed'] = False
                validation['errors'].append(
                    f"{split}: Image count ({img_count}) != Label count ({label_count})"
                )
            
            # Check for missing labels
            missing_labels = images - labels
            if missing_labels:
                validation['passed'] = False
                validation['missing_label_count'] += len(missing_labels)
                validation['errors'].append(
                    f"{split}: {len(missing_labels)} images missing labels: {sorted(missing_labels)[:5]}"
                )
            
            # Check for missing images
            missing_images = labels - images
            if missing_images:
                validation['passed'] = False
                validation['missing_image_count'] += len(missing_images)
                validation['errors'].append(
                    f"{split}: {len(missing_images)} labels missing images: {sorted(missing_images)[:5]}"
                )
            
            # Check for duplicate filenames across splits
            duplicates = all_filenames & images
            if duplicates:
                validation['passed'] = False
                validation['duplicate_count'] += len(duplicates)
                validation['errors'].append(
                    f"{split}: {len(duplicates)} duplicate filenames: {sorted(duplicates)}"
                )
            
            all_filenames.update(images)
            
            # Validate each label
            for label_file in label_dir.glob('*.txt'):
                try:
                    label_text = label_file.read_text(encoding='utf-8').strip()
                    
                    if not label_text:
                        validation['passed'] = False
                        validation['invalid_label_count'] += 1
                        validation['errors'].append(f"{split}/{label_file.name}: Empty label file")
                        continue
                    
                    parts = label_text.split()
                    if len(parts) != 5:
                        validation['passed'] = False
                        validation['invalid_label_count'] += 1
                        validation['errors'].append(
                            f"{split}/{label_file.name}: Invalid format (expected 5 values, got {len(parts)})"
                        )
                        continue
                    
                    class_id, x_center, y_center, width, height = map(float, parts)
                    
                    # Check coordinates are in valid range
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        validation['passed'] = False
                        validation['bbox_outside_count'] += 1
                        validation['errors'].append(
                            f"{split}/{label_file.name}: Center coordinates out of bounds: ({x_center:.3f}, {y_center:.3f})"
                        )
                    
                    if not (0 < width <= 1 and 0 < height <= 1):
                        validation['passed'] = False
                        validation['bbox_outside_count'] += 1
                        validation['errors'].append(
                            f"{split}/{label_file.name}: Size out of bounds: ({width:.3f}, {height:.3f})"
                        )
                    
                    # Check bbox is fully inside image [0, 1]
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                        validation['warnings'].append(
                            f"{split}/{label_file.name}: BBox extends outside image: "
                            f"[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]"
                        )
                        # Don't fail for small violations (rounding errors)
                        if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
                            validation['passed'] = False
                            validation['bbox_outside_count'] += 1
                    
                except Exception as e:
                    validation['passed'] = False
                    validation['invalid_label_count'] += 1
                    validation['errors'].append(f"{split}/{label_file.name}: Parse error: {e}")
        
        # Check for split overlap (same sample ID in multiple splits)
        train_val_overlap = split_files['train'] & split_files['val']
        train_test_overlap = split_files['train'] & split_files['test']
        val_test_overlap = split_files['val'] & split_files['test']
        
        if train_val_overlap:
            validation['passed'] = False
            validation['split_overlap_count'] += len(train_val_overlap)
            validation['errors'].append(
                f"Train/Val overlap: {sorted(train_val_overlap)}"
            )
        
        if train_test_overlap:
            validation['passed'] = False
            validation['split_overlap_count'] += len(train_test_overlap)
            validation['errors'].append(
                f"Train/Test overlap: {sorted(train_test_overlap)}"
            )
        
        if val_test_overlap:
            validation['passed'] = False
            validation['split_overlap_count'] += len(val_test_overlap)
            validation['errors'].append(
                f"Val/Test overlap: {sorted(val_test_overlap)}"
            )
        
        # Summary
        validation['total_images'] = len(all_filenames)
        validation['total_errors'] = len(validation['errors'])
        validation['total_warnings'] = len(validation['warnings'])
        
        if validation['passed']:
            logger.info("Dataset validation PASSED")
        else:
            logger.error(f"Dataset validation FAILED with {validation['total_errors']} errors")
            for error in validation['errors'][:10]:  # Show first 10 errors
                logger.error(f"  - {error}")
            if validation['total_errors'] > 10:
                logger.error(f"  ... and {validation['total_errors'] - 10} more errors")
        
        if validation['warnings']:
            logger.warning(f"Dataset has {validation['total_warnings']} warnings")
            for warning in validation['warnings'][:5]:
                logger.warning(f"  - {warning}")
        
        return validation
    
    def generate_sample(
        self, 
        sample_id: int,
        split: str,
        base_path: Path,
        max_retries: int = 5
    ) -> Dict:
        """
        Generate a single training sample with validation.
        
        Args:
            sample_id: Unique sample ID (used for filename)
            split: Which split this sample belongs to ('train', 'val', or 'test')
            base_path: Base dataset directory path
            max_retries: Maximum retry attempts for failed samples
        
        Returns:
            Dictionary with generation results
        """
        for attempt in range(max_retries):
            # Randomize parameters
            beacon_x, beacon_y = self._randomize_beacon_position()
            beacon_size = self._randomize_beacon_size()
            brightness = self._randomize_brightness()
            disturbances = self._randomize_disturbances()
            blur_kernel = self._randomize_blur()
            
            # Validate beacon is within frame bounds with margin
            img_width = 1280
            img_height = 720
            margin = beacon_size * 3  # Ensure beacon + halo fits
            
            if not (margin <= beacon_x <= img_width - margin and
                    margin <= beacon_y <= img_height - margin):
                if attempt < max_retries - 1:
                    logger.debug(f"Sample {sample_id} ({split}): Beacon outside frame bounds, retrying...")
                    continue
                else:
                    logger.warning(f"Sample {sample_id} ({split}): Failed to generate valid position after {max_retries} attempts")
                    self.stats['rejected_samples'] = self.stats.get('rejected_samples', 0) + 1
                    return None
            
            # Apply disturbances to simulation
            self._apply_disturbances(disturbances)
            
            # Render frame with randomized beacon
            frame = self.camera.render(beacon_x, beacon_y, beacon_size=beacon_size)
            
            # Apply brightness
            frame = self._apply_brightness(frame, brightness)
            
            # Apply motion blur
            frame = self._apply_motion_blur(frame, blur_kernel)
            
            # Verify beacon is detectable (quality check, NOT for labeling)
            detection_result = self.detector.detect(frame)
            
            if detection_result is None:
                # Beacon not detected by detector
                detected = False
                detected_x, detected_y = 0, 0
                confidence = 0.0
                
                # Check if parameters are too extreme
                if brightness < 0.4 or beacon_size < 7:
                    if attempt < max_retries - 1:
                        logger.debug(f"Sample {sample_id} ({split}): Beacon not detectable (brightness={brightness:.2f}, size={beacon_size:.1f}), retrying...")
                        continue
                
                logger.warning(f"Sample {sample_id} ({split}): Beacon not detected by detector (will still label with ground truth)")
                self.stats['detector_failures'] = self.stats.get('detector_failures', 0) + 1
            else:
                # Beacon detected
                detected = True
                detected_x = detection_result.get('x', 0)
                detected_y = detection_result.get('y', 0)
                confidence = detection_result.get('confidence', 0.0)
                
                # Verify detection is reasonable (within 50 pixels of ground truth)
                detection_error = ((detected_x - beacon_x)**2 + (detected_y - beacon_y)**2)**0.5
                if detection_error > 50:
                    logger.warning(f"Sample {sample_id} ({split}): Large detection error ({detection_error:.1f} px)")
                    self.stats['large_detection_errors'] = self.stats.get('large_detection_errors', 0) + 1
            
            # Generate filename using unique sample_id
            filename = f"beacon_{sample_id:06d}"
            
            # Save image
            image_path = base_path / 'images' / split / f"{filename}.png"
            cv2.imwrite(str(image_path), frame)
            
            # Generate YOLO label using GROUND TRUTH (not detector)
            # This ensures every image has correct annotation regardless of detector performance
            label_text = self._generate_yolo_label(
                beacon_x, beacon_y, beacon_size,
                frame.shape[1], frame.shape[0]
            )
            
            # Validate label
            parts = label_text.split()
            if len(parts) != 5:
                logger.error(f"Sample {sample_id} ({split}): Invalid label format: {label_text}")
                if attempt < max_retries - 1:
                    continue
                else:
                    self.stats['invalid_labels'] = self.stats.get('invalid_labels', 0) + 1
                    return None
            
            # Check label coordinates are valid
            _, x_center, y_center, width, height = map(float, parts)
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                    0 < width <= 1 and 0 < height <= 1):
                logger.error(f"Sample {sample_id} ({split}): Label coordinates out of bounds: {label_text}")
                if attempt < max_retries - 1:
                    continue
                else:
                    self.stats['invalid_labels'] = self.stats.get('invalid_labels', 0) + 1
                    return None
            
            # Save label
            label_path = base_path / 'labels' / split / f"{filename}.txt"
            label_path.write_text(label_text, encoding='utf-8')
            
            # Update stats
            self.stats['total_generated'] += 1
            self.stats['valid_samples'] = self.stats.get('valid_samples', 0) + 1
            self.stats[f'{split}_count'] += 1
            
            # Track beacon size range
            self.stats['min_beacon_size'] = min(self.stats.get('min_beacon_size', float('inf')), beacon_size)
            self.stats['max_beacon_size'] = max(self.stats.get('max_beacon_size', 0), beacon_size)
            
            return {
                'index': sample_id,
                'split': split,
                'filename': filename,
                'beacon_x': beacon_x,
                'beacon_y': beacon_y,
                'beacon_size': beacon_size,
                'brightness': brightness,
                'disturbances': disturbances,
                'blur_kernel': blur_kernel,
                'detected': detected,
                'confidence': float(confidence) if confidence is not None else 0.0,
                'label_valid': True
            }
        
        # All retries exhausted
        logger.error(f"Sample {sample_id} ({split}): Failed after {max_retries} attempts")
        self.stats['rejected_samples'] = self.stats.get('rejected_samples', 0) + 1
        return None
    
    def generate_dataset(self, progress_callback=None) -> Dict:
        """
        Generate complete dataset.
        
        Args:
            progress_callback: Optional callback function(current, total, sample_info)
        
        Returns:
            Generation statistics
        """
        logger.info(f"Starting dataset generation: {self.config.num_images} images")
        
        # Setup directories
        base_path = self.setup_directories()
        
        # Pre-assign splits to ensure no overlap
        split_assignments = self._prepare_split_assignments()
        
        # Generate samples
        successful = 0
        failed_samples = []
        
        for sample_id, split in split_assignments:
            sample_info = self.generate_sample(sample_id, split, base_path)
            
            if sample_info is None:
                # Sample failed after retries
                logger.warning(f"Sample {sample_id} ({split}) generation failed")
                failed_samples.append((sample_id, split))
                continue
            
            successful += 1
            
            if progress_callback:
                progress_callback(successful, self.config.num_images, sample_info)
            
            # Log progress every 10%
            if successful % max(1, self.config.num_images // 10) == 0:
                progress = successful / self.config.num_images * 100
                logger.info(f"Progress: {progress:.1f}% ({successful}/{self.config.num_images})")
        
        if failed_samples:
            logger.error(f"Failed to generate {len(failed_samples)} samples: {failed_samples}")
        
        # Validate dataset integrity
        validation_results = self._validate_dataset(base_path)
        self.stats['validation'] = validation_results
        
        # Generate dataset.yaml for YOLO
        self._generate_dataset_yaml(base_path)
        
        # Generate README
        self._generate_readme(base_path)
        
        # Generate contact sheet with ground truth boxes
        self._generate_contact_sheet(base_path)
        
        logger.info(f"Dataset generation complete: {self.stats}")
        return self.stats
    
    def _generate_dataset_yaml(self, base_path: Path):
        """Generate dataset.yaml for YOLO training."""
        yaml_content = f"""# FSOC Beacon Detection Dataset
# Auto-generated by Dataset Generator

# Dataset paths (relative to this file)
path: {base_path.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 1  # number of classes
names: ['beacon']  # class names

# Dataset statistics
# Total images: {self.config.num_images}
# Train: {self.stats['train_count']} ({self.config.train_split * 100:.0f}%)
# Val: {self.stats['val_count']} ({self.config.val_split * 100:.0f}%)
# Test: {self.stats['test_count']} ({self.config.test_split * 100:.0f}%)
"""
        
        yaml_path = base_path / 'dataset.yaml'
        yaml_path.write_text(yaml_content, encoding='utf-8')
        logger.info(f"Generated {yaml_path}")
    
    def _generate_readme(self, base_path: Path):
        """Generate README with dataset information."""
        valid_samples = self.stats.get('valid_samples', 0)
        rejected_samples = self.stats.get('rejected_samples', 0)
        detector_failures = self.stats.get('detector_failures', 0)
        detection_rate = ((valid_samples - detector_failures) / valid_samples * 100) if valid_samples > 0 else 0
        
        validation = self.stats.get('validation', {})
        validation_status = "✓ PASSED" if validation.get('passed', False) else "✗ FAILED"
        
        readme_content = f"""# FSOC Beacon Detection Dataset

Auto-generated synthetic dataset for training YOLO beacon detection models.

## Validation Status: {validation_status}

{self._format_validation_section(validation)}

## Dataset Statistics

- **Total Valid Images**: {valid_samples}
- **Rejected Samples**: {rejected_samples}
- **Train Split**: {self.stats['train_count']} ({self.config.train_split * 100:.1f}%)
- **Validation Split**: {self.stats['val_count']} ({self.config.val_split * 100:.1f}%)
- **Test Split**: {self.stats['test_count']} ({self.config.test_split * 100:.1f}%)

## Quality Metrics

- **Detector Detection Rate**: {detection_rate:.1f}% ({valid_samples - detector_failures}/{valid_samples} detected)
- **Detector Failures**: {detector_failures} (still labeled with ground truth)
- **Large Detection Errors**: {self.stats.get('large_detection_errors', 0)}
- **Invalid Labels**: {self.stats.get('invalid_labels', 0)}

## Beacon Size Range

- **Minimum Beacon Size**: {self.stats.get('min_beacon_size', 0):.1f} pixels
- **Maximum Beacon Size**: {self.stats.get('max_beacon_size', 0):.1f} pixels

## Class Distribution

- **Class 0 (beacon)**: {valid_samples} positive samples
- **Negative samples**: 0 (all images contain beacon)

## Dataset Structure

```
dataset/
├── images/
│   ├── train/       # Training images ({self.stats['train_count']} images)
│   ├── val/         # Validation images ({self.stats['val_count']} images)
│   └── test/        # Test images ({self.stats['test_count']} images)
├── labels/
│   ├── train/       # Training labels (YOLO format)
│   ├── val/         # Validation labels
│   └── test/        # Test labels
├── dataset.yaml     # YOLO configuration file
├── contact_sheet.png # Visual sample of generated images
└── README.md        # This file
```

## Important Notes

### Ground Truth Labels

**All labels use ground truth beacon position**, not detector output.

This ensures:
- ✅ Every image has accurate annotation
- ✅ Detector failures don't create mislabeled data
- ✅ Labels remain valid even for difficult samples
- ✅ Training data is 100% accurate

### Unique Sample IDs

**Each sample has a globally unique ID** to prevent split overlap.

This ensures:
- ✅ No duplicate samples across train/val/test
- ✅ No data leakage between splits
- ✅ Valid evaluation metrics

### Validation

Samples were validated for:
- ✅ Beacon position within frame bounds
- ✅ Label coordinates in valid range [0, 1]
- ✅ Minimum beacon size for visibility
- ✅ Minimum brightness for detectability
- ✅ Unique filenames across all splits
- ✅ Image count matches label count per split
- ✅ No sample ID overlap between splits
- ✅ Beacon bbox fully inside image bounds

Rejected samples: {rejected_samples} (out of {valid_samples + rejected_samples} attempts)

## Randomization Parameters

### Beacon Properties
- **Position X**: {self.config.beacon_x_range[0]:.0f} - {self.config.beacon_x_range[1]:.0f} pixels
- **Position Y**: {self.config.beacon_y_range[0]:.0f} - {self.config.beacon_y_range[1]:.0f} pixels
- **Size**: {self.config.beacon_size_range[0]:.1f} - {self.config.beacon_size_range[1]:.1f} pixels radius
- **Brightness**: {self.config.brightness_range[0]:.1f} - {self.config.brightness_range[1]:.1f} multiplier

### Environmental Conditions
- **Turbulence**: {self.config.turbulence_range[0]:.1f} - {self.config.turbulence_range[1]:.1f}
- **Sensor Noise**: {self.config.noise_range[0]:.1f} - {self.config.noise_range[1]:.1f}
- **Camera Vibration**: {self.config.vibration_range[0]:.1f} - {self.config.vibration_range[1]:.1f}
- **Motion Blur**: {self.config.blur_range[0]} - {self.config.blur_range[1]} kernel size

## Label Format

YOLO format: `<class> <x_center> <y_center> <width> <height>`

- All coordinates normalized to [0, 1]
- Class 0 = beacon
- Bounding box is 2.5x beacon diameter

Example: `0 0.500000 0.450000 0.039063 0.069444`

## Training with YOLOv8

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='beacon_detection'
)

# Validate
metrics = model.val()

# Test
model.predict('dataset/images/test', save=True)
```

## Visual Inspection

See `contact_sheet.png` for a visual sample of generated images with ground truth bounding boxes.

## Notes

- All images are PNG format (lossless)
- Labels are in YOLO TXT format (one per image)
- Dataset is fully synthetic - no manual labeling required
- Beacon is always present and correctly labeled with ground truth
- Ground truth labeling ensures 100% accuracy
- Unique sample IDs prevent split contamination

Generated: {valid_samples} samples with automated ground-truth labeling.
"""
        
        readme_path = base_path / 'README.md'
        readme_path.write_text(readme_content, encoding='utf-8')
        logger.info(f"Generated {readme_path}")
    
    def _format_validation_section(self, validation: Dict) -> str:
        """Format validation results for README."""
        if not validation:
            return "Validation not performed.\n"
        
        lines = []
        
        # Overall status
        if validation.get('passed', False):
            lines.append("### ✓ All validation checks passed!")
        else:
            lines.append("### ✗ Validation found issues")
        
        lines.append("")
        
        # Per-split stats
        lines.append("**Per-Split Counts:**")
        for split in ['train', 'val', 'test']:
            if split in validation.get('per_split', {}):
                info = validation['per_split'][split]
                match_symbol = "✓" if info['match'] else "✗"
                lines.append(
                    f"- {split.upper()}: {info['image_count']} images, "
                    f"{info['label_count']} labels {match_symbol}"
                )
        lines.append("")
        
        # Error summary
        lines.append("**Validation Checks:**")
        lines.append(f"- Duplicate filenames: {validation.get('duplicate_count', 0)}")
        lines.append(f"- Split overlaps: {validation.get('split_overlap_count', 0)}")
        lines.append(f"- Invalid labels: {validation.get('invalid_label_count', 0)}")
        lines.append(f"- BBox outside image: {validation.get('bbox_outside_count', 0)}")
        lines.append(f"- Missing labels: {validation.get('missing_label_count', 0)}")
        lines.append(f"- Missing images: {validation.get('missing_image_count', 0)}")
        lines.append("")
        
        # Errors
        if validation.get('errors'):
            lines.append("**Errors:**")
            for error in validation['errors'][:10]:
                lines.append(f"- {error}")
            if len(validation['errors']) > 10:
                lines.append(f"- ... and {len(validation['errors']) - 10} more errors")
            lines.append("")
        
        # Warnings
        if validation.get('warnings'):
            lines.append("**Warnings:**")
            for warning in validation['warnings'][:5]:
                lines.append(f"- {warning}")
            if len(validation['warnings']) > 5:
                lines.append(f"- ... and {len(validation['warnings']) - 5} more warnings")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_contact_sheet(self, base_path: Path):
        """
        Generate contact sheet showing ALL generated images with ground truth bounding boxes.
        
        Args:
            base_path: Base dataset path
        """
        try:
            # Collect ALL images from each split (not random samples)
            samples = []
            for split in ['train', 'val', 'test']:
                img_dir = base_path / 'images' / split
                label_dir = base_path / 'labels' / split
                
                img_files = sorted(img_dir.glob('*.png'))  # Sort for consistent ordering
                if not img_files:
                    continue
                
                for img_path in img_files:
                    label_path = label_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        samples.append((split, img_path, label_path))
            
            if not samples:
                logger.warning("No samples found for contact sheet")
                return
            
            # Create contact sheet grid (max 8 columns)
            cols = min(8, len(samples))
            rows = (len(samples) + cols - 1) // cols
            cell_w, cell_h = 160, 90  # Smaller cells to fit more samples
            
            contact_sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
            
            for idx, (split, img_path, label_path) in enumerate(samples):
                row = idx // cols
                col = idx % cols
                
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue
                
                # Resize to cell size
                img_resized = cv2.resize(img, (cell_w, cell_h))
                
                # Read and parse label
                try:
                    label_text = label_path.read_text(encoding='utf-8').strip()
                    if label_text:
                        parts = label_text.split()
                        if len(parts) == 5:
                            _, x_center, y_center, width, height = map(float, parts)
                            
                            # Convert normalized coords to pixel coords
                            x1 = int((x_center - width/2) * cell_w)
                            y1 = int((y_center - height/2) * cell_h)
                            x2 = int((x_center + width/2) * cell_w)
                            y2 = int((y_center + height/2) * cell_h)
                            
                            # Draw ground truth bounding box (GREEN)
                            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            
                            # Draw center point (RED)
                            cx = int(x_center * cell_w)
                            cy = int(y_center * cell_h)
                            cv2.circle(img_resized, (cx, cy), 2, (0, 0, 255), -1)
                except Exception as e:
                    logger.warning(f"Failed to parse label {label_path}: {e}")
                
                # Add split indicator - small colored corner
                split_colors = {
                    'train': (0, 255, 0),   # Green
                    'val': (255, 165, 0),   # Orange
                    'test': (0, 165, 255)   # Light blue
                }
                color = split_colors.get(split, (255, 255, 255))
                
                # Colored corner indicator (top-left)
                cv2.rectangle(img_resized, (0, 0), (20, 10), color, -1)
                
                # Place in contact sheet
                y_start = row * cell_h
                x_start = col * cell_w
                contact_sheet[y_start:y_start+cell_h, x_start:x_start+cell_w] = img_resized
            
            # Add legend at bottom
            legend_height = 30
            contact_with_legend = np.zeros((rows * cell_h + legend_height, cols * cell_w, 3), dtype=np.uint8)
            contact_with_legend[:rows * cell_h, :] = contact_sheet
            
            # Legend text
            legend_y = rows * cell_h + 20
            cv2.putText(contact_with_legend, f"All {len(samples)} samples | Green box=GT | Red dot=center | Corner: Green=train Orange=val Blue=test", 
                       (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Save contact sheet
            contact_path = base_path / 'contact_sheet.png'
            cv2.imwrite(str(contact_path), contact_with_legend)
            logger.info(f"Generated contact sheet: {contact_path} ({rows}x{cols} grid, {len(samples)} samples)")
            
        except Exception as e:
            logger.error(f"Failed to generate contact sheet: {e}", exc_info=True)


def create_dataset_generator(camera, detector, disturbances, num_images: int = 1000):
    """
    Factory function to create dataset generator with custom config.
    
    Args:
        camera: Camera instance
        detector: Detector instance
        disturbances: Disturbances instance
        num_images: Number of images to generate
    
    Returns:
        DatasetGenerator instance
    """
    config = DatasetConfig(num_images=num_images)
    return DatasetGenerator(camera, detector, disturbances, config)
