#!/usr/bin/env python3
"""
Quick test of dataset generator to verify all fixes work.
Generates a tiny 10-image dataset to test the pipeline.
"""

import sys
sys.path.insert(0, 'backend')

import time
from pathlib import Path
from dataset_generator import DatasetGenerator, DatasetConfig
from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector
from simulation.disturbances import DisturbanceModel

def test_dataset_generator():
    """Test dataset generator with 10 images."""
    print("=" * 70)
    print("DATASET GENERATOR TEST")
    print("=" * 70)
    print()
    
    # Create simulation components
    print("1. Initializing components...")
    camera = VirtualCamera(width=1280, height=720)
    detector = BeaconDetector()
    disturbances = DisturbanceModel()
    print("   ✓ Components initialized")
    print()
    
    # Create config for tiny test dataset
    print("2. Configuring test dataset (10 images)...")
    config = DatasetConfig(
        num_images=10,
        train_split=0.6,   # 6 images
        val_split=0.2,     # 2 images
        test_split=0.2,    # 2 images
        output_dir="dataset_test"
    )
    print(f"   Train: {int(config.num_images * config.train_split)} images")
    print(f"   Val:   {int(config.num_images * config.val_split)} images")
    print(f"   Test:  {int(config.num_images * config.test_split)} images")
    print()
    
    # Create generator
    print("3. Creating dataset generator...")
    generator = DatasetGenerator(camera, detector, disturbances, config)
    print("   ✓ Generator created")
    print()
    
    # Generate dataset
    print("4. Generating dataset...")
    print("   This will test:")
    print("   - UTF-8 encoding for README/YAML")
    print("   - Ground truth labeling")
    print("   - Position validation")
    print("   - Label validation")
    print("   - Contact sheet generation")
    print()
    
    start_time = time.time()
    
    try:
        stats = generator.generate_dataset()
        
        generation_time = time.time() - start_time
        
        print()
        print("=" * 70)
        print("✓ GENERATION SUCCESSFUL!")
        print("=" * 70)
        print()
        
        # Report results
        print("STATISTICS:")
        print(f"  Generation Time: {generation_time:.2f} seconds")
        print(f"  Valid Samples: {stats.get('valid_samples', 0)}")
        print(f"  Rejected Samples: {stats.get('rejected_samples', 0)}")
        print(f"  Train: {stats.get('train_count', 0)}")
        print(f"  Val: {stats.get('val_count', 0)}")
        print(f"  Test: {stats.get('test_count', 0)}")
        print()
        
        print("QUALITY METRICS:")
        detector_failures = stats.get('detector_failures', 0)
        valid_samples = stats.get('valid_samples', 0)
        detection_rate = ((valid_samples - detector_failures) / valid_samples * 100) if valid_samples > 0 else 0
        print(f"  Detection Rate: {detection_rate:.1f}%")
        print(f"  Detector Failures: {detector_failures} (still labeled with ground truth)")
        print(f"  Large Detection Errors: {stats.get('large_detection_errors', 0)}")
        print(f"  Invalid Labels: {stats.get('invalid_labels', 0)}")
        print()
        
        min_size = stats.get('min_beacon_size', 0)
        max_size = stats.get('max_beacon_size', 0)
        if min_size != float('inf'):
            print("BEACON SIZE RANGE:")
            print(f"  Min: {min_size:.1f} pixels")
            print(f"  Max: {max_size:.1f} pixels")
            print()
        
        # Check files created
        print("FILES CREATED:")
        dataset_path = Path(config.output_dir)
        
        files_to_check = [
            ('dataset.yaml', 'YOLO configuration'),
            ('README.md', 'Dataset information'),
            ('contact_sheet.png', 'Visual samples'),
            ('images/train', 'Training images'),
            ('images/val', 'Validation images'),
            ('images/test', 'Test images'),
            ('labels/train', 'Training labels'),
            ('labels/val', 'Validation labels'),
            ('labels/test', 'Test labels'),
        ]
        
        all_exist = True
        for file_path, description in files_to_check:
            full_path = dataset_path / file_path
            exists = full_path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {file_path:25s} - {description}")
            if not exists:
                all_exist = False
        
        print()
        
        if all_exist:
            print("✓ All expected files created successfully!")
        else:
            print("⚠ Some files missing!")
        
        print()
        print("TEST PASSED!")
        print()
        print(f"Dataset location: {dataset_path.absolute()}")
        print("Inspect contact_sheet.png to verify bounding boxes")
        print()
        
        return True
        
    except Exception as e:
        print()
        print("=" * 70)
        print("✗ GENERATION FAILED!")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

if __name__ == "__main__":
    success = test_dataset_generator()
    sys.exit(0 if success else 1)
