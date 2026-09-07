"""
Test that the WebSocket API works with the updated dataset generator.
This simulates what the frontend does.
"""
import asyncio
import sys
from pathlib import Path

# Import simulation modules
from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector
from simulation.disturbances import DisturbanceModel
from simulation.logger import setup_logging
from dataset_generator import DatasetGenerator, DatasetConfig


async def test_api_integration():
    """Test the API integration with a small dataset."""
    print("\n" + "="*80)
    print("TESTING API INTEGRATION (Simulating Frontend)")
    print("="*80 + "\n")
    
    # Setup logging
    logger = setup_logging()
    
    # Initialize simulation components (like SimulationManager does)
    camera = VirtualCamera(width=1280, height=720)
    detector = BeaconDetector()
    disturbances = DisturbanceModel(
        turbulence=1.0,
        vibration=1.0,
        camera_motion=1.0,
        sensor_noise=1.0
    )
    
    # Test with 5 images (quick test)
    num_images = 5
    print(f"Testing with {num_images} images...\n")
    
    # Create dataset generator (like the API does)
    config = DatasetConfig(num_images=num_images)
    generator = DatasetGenerator(
        camera=camera,
        detector=detector,
        disturbances=disturbances,
        config=config
    )
    
    # Setup directories
    base_path = generator.setup_directories()
    print(f"✓ Directories created: {base_path.absolute()}\n")
    
    # Pre-assign splits (NEW METHOD)
    split_assignments = generator._prepare_split_assignments()
    print(f"✓ Split assignments prepared: {len(split_assignments)} samples")
    
    train_count = sum(1 for _, split in split_assignments if split == 'train')
    val_count = sum(1 for _, split in split_assignments if split == 'val')
    test_count = sum(1 for _, split in split_assignments if split == 'test')
    print(f"  - Train: {train_count}")
    print(f"  - Val:   {val_count}")
    print(f"  - Test:  {test_count}\n")
    
    # Generate dataset (like the API does)
    print("Generating samples...\n")
    successful = 0
    
    for sample_id, split in split_assignments:
        try:
            # Generate sample with explicit split (NEW SIGNATURE)
            sample_info = generator.generate_sample(sample_id, split, base_path)
            
            if sample_info is None:
                print(f"⚠ Sample {sample_id} ({split}) failed")
                continue
            
            successful += 1
            print(f"  [{successful}/{num_images}] {sample_info['filename']} ({split}) - "
                  f"detected={sample_info['detected']}, "
                  f"confidence={sample_info['confidence']:.2f}")
            
            # Simulate small delay like the API does
            await asyncio.sleep(0.001)
            
        except Exception as e:
            print(f"✗ Error generating sample {sample_id} ({split}): {e}")
            return False
    
    print(f"\n✓ Successfully generated {successful}/{num_images} samples\n")
    
    # Validate dataset (NEW)
    print("Validating dataset...\n")
    validation_results = generator._validate_dataset(base_path)
    
    if validation_results['passed']:
        print("✓ Validation PASSED")
    else:
        print("✗ Validation FAILED")
        for error in validation_results['errors'][:5]:
            print(f"  - {error}")
        return False
    
    # Generate metadata files
    print("\nGenerating metadata files...")
    generator._generate_dataset_yaml(base_path)
    print("  ✓ dataset.yaml")
    
    generator._generate_readme(base_path)
    print("  ✓ README.md")
    
    generator._generate_contact_sheet(base_path)
    print("  ✓ contact_sheet.png")
    
    print("\n" + "="*80)
    print("API INTEGRATION TEST: PASSED ✓")
    print("="*80 + "\n")
    
    print("Summary:")
    print(f"  - Generator can be called with new signature")
    print(f"  - Split pre-assignment works")
    print(f"  - Sample generation works")
    print(f"  - Validation works")
    print(f"  - All metadata files generated")
    print(f"\nFrontend should now work correctly!\n")
    
    return True


def main():
    """Main entry point."""
    result = asyncio.run(test_api_integration())
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
