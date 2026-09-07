"""
Dataset Verification Script

Validates that the generated dataset meets all requirements:
- Unique filenames across train/val/test
- Image count == label count per split
- No sample ID overlap between splits
- Every label is valid YOLO format
- Beacon bbox fully inside image bounds
"""
import sys
from pathlib import Path
from collections import defaultdict

def verify_dataset(dataset_dir='dataset'):
    """Verify dataset integrity."""
    base_path = Path(dataset_dir)
    
    if not base_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    print(f"\n{'='*70}")
    print(f"DATASET VERIFICATION: {base_path.absolute()}")
    print(f"{'='*70}\n")
    
    all_passed = True
    all_files = defaultdict(list)
    
    # Check each split
    for split in ['train', 'val', 'test']:
        img_dir = base_path / 'images' / split
        label_dir = base_path / 'labels' / split
        
        print(f"📁 {split.upper()} Split:")
        
        # Get files
        images = sorted([p.stem for p in img_dir.glob('*.png')])
        labels = sorted([p.stem for p in label_dir.glob('*.txt')])
        
        print(f"   Images: {len(images)}")
        print(f"   Labels: {len(labels)}")
        
        # Store for overlap check
        all_files[split] = set(images)
        
        # Check 1: Image count == Label count
        if len(images) == len(labels):
            print(f"   ✓ Image count matches label count")
        else:
            print(f"   ❌ Image count ({len(images)}) != Label count ({len(labels)})")
            all_passed = False
        
        # Check 2: Missing labels
        missing_labels = set(images) - set(labels)
        if not missing_labels:
            print(f"   ✓ No missing labels")
        else:
            print(f"   ❌ Missing labels: {missing_labels}")
            all_passed = False
        
        # Check 3: Missing images
        missing_images = set(labels) - set(images)
        if not missing_images:
            print(f"   ✓ No missing images")
        else:
            print(f"   ❌ Missing images: {missing_images}")
            all_passed = False
        
        # Check 4: Validate label format
        invalid_labels = []
        bbox_errors = []
        
        for label_file in label_dir.glob('*.txt'):
            try:
                content = label_file.read_text(encoding='utf-8').strip()
                if not content:
                    invalid_labels.append(f"{label_file.name}: Empty")
                    continue
                
                parts = content.split()
                if len(parts) != 5:
                    invalid_labels.append(f"{label_file.name}: Wrong format (expected 5 values)")
                    continue
                
                class_id, x_center, y_center, width, height = map(float, parts)
                
                # Check coordinates are in valid range [0, 1]
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    invalid_labels.append(f"{label_file.name}: Center out of bounds")
                    continue
                
                if not (0 < width <= 1 and 0 < height <= 1):
                    invalid_labels.append(f"{label_file.name}: Size out of bounds")
                    continue
                
                # Check bbox is fully inside image
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
                    bbox_errors.append(f"{label_file.name}: BBox outside image")
                
            except Exception as e:
                invalid_labels.append(f"{label_file.name}: Parse error: {e}")
        
        if not invalid_labels:
            print(f"   ✓ All labels valid YOLO format")
        else:
            print(f"   ❌ Invalid labels found:")
            for error in invalid_labels[:5]:
                print(f"      - {error}")
            if len(invalid_labels) > 5:
                print(f"      ... and {len(invalid_labels) - 5} more")
            all_passed = False
        
        if not bbox_errors:
            print(f"   ✓ All bounding boxes inside image bounds")
        else:
            print(f"   ❌ BBox errors:")
            for error in bbox_errors[:5]:
                print(f"      - {error}")
            all_passed = False
        
        print()
    
    # Check 5: No overlap between splits
    print(f"🔍 Cross-Split Validation:")
    
    train_val = all_files['train'] & all_files['val']
    train_test = all_files['train'] & all_files['test']
    val_test = all_files['val'] & all_files['test']
    
    overlap_found = False
    
    if not train_val:
        print(f"   ✓ No train-val overlap")
    else:
        print(f"   ❌ Train-Val overlap: {sorted(train_val)}")
        overlap_found = True
        all_passed = False
    
    if not train_test:
        print(f"   ✓ No train-test overlap")
    else:
        print(f"   ❌ Train-Test overlap: {sorted(train_test)}")
        overlap_found = True
        all_passed = False
    
    if not val_test:
        print(f"   ✓ No val-test overlap")
    else:
        print(f"   ❌ Val-Test overlap: {sorted(val_test)}")
        overlap_found = True
        all_passed = False
    
    # Check 6: Unique filenames globally
    all_sample_ids = list(all_files['train']) + list(all_files['val']) + list(all_files['test'])
    unique_count = len(set(all_sample_ids))
    total_count = len(all_sample_ids)
    
    if unique_count == total_count:
        print(f"   ✓ All {total_count} sample IDs are unique")
    else:
        print(f"   ❌ Duplicate sample IDs found: {total_count - unique_count} duplicates")
        all_passed = False
    
    print()
    
    # Summary
    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total Images: {len(all_sample_ids)}")
    print(f"  Train: {len(all_files['train'])}")
    print(f"  Val:   {len(all_files['val'])}")
    print(f"  Test:  {len(all_files['test'])}")
    print(f"\nValidation: {'✓ PASSED' if all_passed else '❌ FAILED'}")
    print(f"{'='*70}\n")
    
    return all_passed


def main():
    """Main entry point."""
    dataset_dir = 'dataset'
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    
    passed = verify_dataset(dataset_dir)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
