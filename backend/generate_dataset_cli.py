"""
CLI script to generate dataset.
"""
import sys
import logging
from pathlib import Path

# Import existing simulation modules
from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector
from simulation.disturbances import DisturbanceModel
from simulation.logger import setup_logging
from dataset_generator import DatasetGenerator, DatasetConfig

def main():
    """Generate dataset from command line."""
    # Setup logging
    logger = setup_logging()
    
    # Get number of images from command line
    num_images = 10  # Default
    if len(sys.argv) > 1:
        try:
            num_images = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python {sys.argv[0]} [num_images]")
            print(f"Invalid number: {sys.argv[1]}")
            sys.exit(1)
    
    logger.info(f"Generating dataset with {num_images} images")
    
    # Initialize simulation components
    camera = VirtualCamera(width=1280, height=720)
    detector = BeaconDetector()
    disturbances = DisturbanceModel(
        turbulence=1.0,
        vibration=1.0,
        camera_motion=1.0,
        sensor_noise=1.0
    )
    
    # Configure dataset generation
    config = DatasetConfig(num_images=num_images)
    generator = DatasetGenerator(camera, detector, disturbances, config)
    
    # Progress callback
    def progress_callback(current, total, sample_info):
        if current % max(1, total // 10) == 0 or current == total:
            progress = current / total * 100
            split = sample_info.get('split', '?')
            filename = sample_info.get('filename', '?')
            print(f"Progress: {progress:5.1f}% ({current:3}/{total:3}) - {split:5s} {filename}")
    
    # Generate dataset
    print(f"\n{'='*60}")
    print(f"DATASET GENERATION")
    print(f"{'='*60}")
    print(f"Target: {num_images} images")
    print(f"Splits: {config.train_split*100:.0f}% train, {config.val_split*100:.0f}% val, {config.test_split*100:.0f}% test")
    print(f"Output: {config.output_dir}/")
    print(f"{'='*60}\n")
    
    stats = generator.generate_dataset(progress_callback=progress_callback)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total generated: {stats['total_generated']}")
    print(f"  Train: {stats['train_count']}")
    print(f"  Val:   {stats['val_count']}")
    print(f"  Test:  {stats['test_count']}")
    print(f"\nQuality:")
    print(f"  Valid samples: {stats.get('valid_samples', 0)}")
    print(f"  Rejected: {stats.get('rejected_samples', 0)}")
    print(f"  Detector failures: {stats.get('detector_failures', 0)}")
    print(f"  Invalid labels: {stats.get('invalid_labels', 0)}")
    
    # Print validation results
    validation = stats.get('validation', {})
    if validation:
        print(f"\n{'='*60}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*60}")
        status = "✓ PASSED" if validation.get('passed', False) else "✗ FAILED"
        print(f"Status: {status}")
        print(f"\nChecks:")
        print(f"  Duplicate filenames: {validation.get('duplicate_count', 0)}")
        print(f"  Split overlaps: {validation.get('split_overlap_count', 0)}")
        print(f"  Invalid labels: {validation.get('invalid_label_count', 0)}")
        print(f"  BBox outside image: {validation.get('bbox_outside_count', 0)}")
        print(f"  Missing labels: {validation.get('missing_label_count', 0)}")
        print(f"  Missing images: {validation.get('missing_image_count', 0)}")
        
        print(f"\nPer-Split:")
        for split in ['train', 'val', 'test']:
            if split in validation.get('per_split', {}):
                info = validation['per_split'][split]
                match = "✓" if info['match'] else "✗"
                print(f"  {split.upper():5s}: {info['image_count']} images, {info['label_count']} labels {match}")
        
        if validation.get('errors'):
            print(f"\nErrors ({len(validation['errors'])}):")
            for error in validation['errors'][:10]:
                print(f"  - {error}")
            if len(validation['errors']) > 10:
                print(f"  ... and {len(validation['errors']) - 10} more")
        
        if validation.get('warnings'):
            print(f"\nWarnings ({len(validation['warnings'])}):")
            for warning in validation['warnings'][:5]:
                print(f"  - {warning}")
            if len(validation['warnings']) > 5:
                print(f"  ... and {len(validation['warnings']) - 5} more")
    
    print(f"\n{'='*60}")
    print(f"Files generated:")
    print(f"  {config.output_dir}/images/{{train,val,test}}/*.png")
    print(f"  {config.output_dir}/labels/{{train,val,test}}/*.txt")
    print(f"  {config.output_dir}/dataset.yaml")
    print(f"  {config.output_dir}/README.md")
    print(f"  {config.output_dir}/contact_sheet.png")
    print(f"{'='*60}\n")
    
    # Exit with validation status
    if validation and not validation.get('passed', False):
        print("⚠️  Dataset validation FAILED!")
        sys.exit(1)
    else:
        print("✓ Dataset generation successful!")
        sys.exit(0)

if __name__ == "__main__":
    main()
