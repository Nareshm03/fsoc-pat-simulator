"""
Comprehensive Dataset Integrity Test

This script runs all validation checks and generates a detailed report
proving the dataset has no split collisions or integrity issues.
"""
import sys
from pathlib import Path
from collections import Counter


def test_dataset_integrity(dataset_dir='dataset'):
    """
    Run comprehensive integrity tests on the dataset.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    base_path = Path(dataset_dir)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DATASET INTEGRITY TEST")
    print("="*80 + "\n")
    
    if not base_path.exists():
        print(f"❌ FATAL: Dataset directory not found: {dataset_dir}")
        return False
    
    all_tests_passed = True
    test_results = []
    
    # Collect all samples
    splits_data = {}
    all_sample_ids = []
    
    for split in ['train', 'val', 'test']:
        img_dir = base_path / 'images' / split
        label_dir = base_path / 'labels' / split
        
        images = set(p.stem for p in img_dir.glob('*.png'))
        labels = set(p.stem for p in label_dir.glob('*.txt'))
        
        splits_data[split] = {
            'images': images,
            'labels': labels,
            'image_count': len(images),
            'label_count': len(labels)
        }
        
        all_sample_ids.extend(images)
    
    # TEST 1: Unique filenames globally
    print("TEST 1: Globally Unique Sample IDs")
    print("-" * 80)
    
    sample_counts = Counter(all_sample_ids)
    duplicates = {sid: count for sid, count in sample_counts.items() if count > 1}
    
    if not duplicates:
        print(f"✓ PASS: All {len(all_sample_ids)} sample IDs are unique")
        test_results.append(("Unique IDs", True, f"{len(all_sample_ids)} unique"))
    else:
        print(f"❌ FAIL: Found {len(duplicates)} duplicate sample IDs:")
        for sid, count in duplicates.items():
            print(f"   - {sid}: appears {count} times")
        test_results.append(("Unique IDs", False, f"{len(duplicates)} duplicates"))
        all_tests_passed = False
    
    print()
    
    # TEST 2: No split overlap
    print("TEST 2: No Sample ID Overlap Between Splits")
    print("-" * 80)
    
    train_set = splits_data['train']['images']
    val_set = splits_data['val']['images']
    test_set = splits_data['test']['images']
    
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    overlap_count = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    
    if train_val_overlap:
        print(f"❌ FAIL: Train-Val overlap: {sorted(train_val_overlap)}")
        all_tests_passed = False
    else:
        print(f"✓ PASS: No Train-Val overlap")
    
    if train_test_overlap:
        print(f"❌ FAIL: Train-Test overlap: {sorted(train_test_overlap)}")
        all_tests_passed = False
    else:
        print(f"✓ PASS: No Train-Test overlap")
    
    if val_test_overlap:
        print(f"❌ FAIL: Val-Test overlap: {sorted(val_test_overlap)}")
        all_tests_passed = False
    else:
        print(f"✓ PASS: No Val-Test overlap")
    
    test_results.append(("Split Overlap", overlap_count == 0, f"{overlap_count} overlaps"))
    print()
    
    # TEST 3: Image-Label correspondence
    print("TEST 3: Image-Label Count Match Per Split")
    print("-" * 80)
    
    mismatch_count = 0
    for split in ['train', 'val', 'test']:
        img_count = splits_data[split]['image_count']
        lbl_count = splits_data[split]['label_count']
        
        if img_count == lbl_count:
            print(f"✓ PASS: {split.upper()}: {img_count} images = {lbl_count} labels")
        else:
            print(f"❌ FAIL: {split.upper()}: {img_count} images ≠ {lbl_count} labels")
            mismatch_count += 1
            all_tests_passed = False
    
    test_results.append(("Image-Label Match", mismatch_count == 0, f"{mismatch_count} mismatches"))
    print()
    
    # TEST 4: YOLO label format validation
    print("TEST 4: YOLO Label Format Validation")
    print("-" * 80)
    
    invalid_count = 0
    total_labels = 0
    
    for split in ['train', 'val', 'test']:
        label_dir = base_path / 'labels' / split
        
        for label_file in label_dir.glob('*.txt'):
            total_labels += 1
            try:
                content = label_file.read_text(encoding='utf-8').strip()
                
                if not content:
                    print(f"❌ FAIL: {split}/{label_file.name}: Empty label")
                    invalid_count += 1
                    continue
                
                parts = content.split()
                if len(parts) != 5:
                    print(f"❌ FAIL: {split}/{label_file.name}: Expected 5 values, got {len(parts)}")
                    invalid_count += 1
                    continue
                
                class_id, x_center, y_center, width, height = map(float, parts)
                
                # Validate ranges
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    print(f"❌ FAIL: {split}/{label_file.name}: Center out of [0,1]")
                    invalid_count += 1
                    continue
                
                if not (0 < width <= 1 and 0 < height <= 1):
                    print(f"❌ FAIL: {split}/{label_file.name}: Size out of (0,1]")
                    invalid_count += 1
                    continue
                
            except Exception as e:
                print(f"❌ FAIL: {split}/{label_file.name}: Parse error: {e}")
                invalid_count += 1
    
    if invalid_count == 0:
        print(f"✓ PASS: All {total_labels} labels are valid YOLO format")
    else:
        print(f"❌ FAIL: {invalid_count}/{total_labels} labels are invalid")
        all_tests_passed = False
    
    test_results.append(("Label Format", invalid_count == 0, f"{invalid_count}/{total_labels} invalid"))
    print()
    
    # TEST 5: Bounding box bounds check
    print("TEST 5: Bounding Box Inside Image Bounds")
    print("-" * 80)
    
    bbox_errors = 0
    
    for split in ['train', 'val', 'test']:
        label_dir = base_path / 'labels' / split
        
        for label_file in label_dir.glob('*.txt'):
            try:
                content = label_file.read_text(encoding='utf-8').strip()
                parts = content.split()
                
                if len(parts) == 5:
                    _, x_center, y_center, width, height = map(float, parts)
                    
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    # Allow small tolerance for rounding
                    if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
                        print(f"❌ FAIL: {split}/{label_file.name}: BBox outside [0,1]: "
                              f"[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                        bbox_errors += 1
            except:
                pass  # Already caught in format validation
    
    if bbox_errors == 0:
        print(f"✓ PASS: All bounding boxes are inside image bounds")
    else:
        print(f"❌ FAIL: {bbox_errors} bounding boxes extend outside image")
        all_tests_passed = False
    
    test_results.append(("BBox Bounds", bbox_errors == 0, f"{bbox_errors} out of bounds"))
    print()
    
    # TEST 6: File structure
    print("TEST 6: Complete File Structure")
    print("-" * 80)
    
    required_files = [
        'dataset.yaml',
        'README.md',
        'contact_sheet.png'
    ]
    
    missing_files = []
    for filename in required_files:
        if not (base_path / filename).exists():
            missing_files.append(filename)
            print(f"❌ FAIL: Missing required file: {filename}")
        else:
            print(f"✓ PASS: Found {filename}")
    
    structure_ok = len(missing_files) == 0
    test_results.append(("File Structure", structure_ok, f"{len(missing_files)} missing"))
    
    if not structure_ok:
        all_tests_passed = False
    
    print()
    
    # FINAL SUMMARY
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\nDataset: {base_path.absolute()}")
    print(f"\nTotal Samples: {len(all_sample_ids)}")
    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")
    print(f"  Test:  {len(test_set)}")
    print(f"\nTests Run: {len(test_results)}")
    
    passed_tests = sum(1 for _, passed, _ in test_results if passed)
    failed_tests = len(test_results) - passed_tests
    
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {failed_tests}")
    
    print(f"\nDetailed Results:")
    for test_name, passed, details in test_results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status} | {test_name:25s} | {details}")
    
    print(f"\n{'='*80}")
    if all_tests_passed:
        print("OVERALL: ✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nDataset integrity verified:")
        print("  ✓ No duplicate sample IDs")
        print("  ✓ Zero split overlap")
        print("  ✓ All labels valid YOLO format")
        print("  ✓ All bounding boxes within bounds")
        print("  ✓ Complete file structure")
    else:
        print("OVERALL: ❌❌❌ SOME TESTS FAILED ❌❌❌")
        print("\nPlease review the failed tests above and regenerate the dataset.")
    print("="*80 + "\n")
    
    return all_tests_passed


def main():
    """Main entry point."""
    dataset_dir = 'dataset'
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    
    passed = test_dataset_integrity(dataset_dir)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
