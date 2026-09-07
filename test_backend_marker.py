"""
Add a temporary visible marker to backend camera to verify browser displays actual backend image.
This temporarily modifies backend/simulation/camera.py to add a test pattern.
"""
import sys
from pathlib import Path


def add_test_marker():
    """Add a temporary red rectangle test marker to camera rendering."""
    camera_file = Path("backend/simulation/camera.py")
    
    if not camera_file.exists():
        print(f"❌ Camera file not found: {camera_file}")
        return False
    
    content = camera_file.read_text()
    
    # Check if marker already exists
    if "TEST_MARKER_VERIFICATION" in content:
        print("⚠ Test marker already exists in camera.py")
        return True
    
    # Find the return statement in render method
    marker_code = '''
        # TEST_MARKER_VERIFICATION - Temporary marker for validation
        # Draw a red rectangle in top-left corner
        cv2.rectangle(frame, (10, 10), (150, 60), (0, 0, 255), 3)
        cv2.putText(frame, "BACKEND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)
        # END_TEST_MARKER
'''
    
    # Insert before the return statement
    modified_content = content.replace(
        "        return frame",
        marker_code + "\n        return frame"
    )
    
    if modified_content == content:
        print("❌ Could not find insertion point in camera.py")
        return False
    
    # Write modified content
    camera_file.write_text(modified_content)
    print("✅ Test marker ADDED to backend/simulation/camera.py")
    print("   Red rectangle with 'BACKEND' text in top-left corner")
    print("   Restart backend and refresh browser to see it")
    return True


def remove_test_marker():
    """Remove the temporary test marker from camera rendering."""
    camera_file = Path("backend/simulation/camera.py")
    
    if not camera_file.exists():
        print(f"❌ Camera file not found: {camera_file}")
        return False
    
    content = camera_file.read_text()
    
    # Check if marker exists
    if "TEST_MARKER_VERIFICATION" not in content:
        print("⚠ No test marker found in camera.py")
        return True
    
    # Remove everything between markers
    import re
    pattern = r'\s*# TEST_MARKER_VERIFICATION.*?# END_TEST_MARKER\s*'
    modified_content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Write modified content
    camera_file.write_text(modified_content)
    print("✅ Test marker REMOVED from backend/simulation/camera.py")
    print("   Restart backend and refresh browser to verify removal")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "remove":
        success = remove_test_marker()
    else:
        print("="*60)
        print("BACKEND IMAGE VERIFICATION TEST")
        print("="*60)
        print()
        print("This will temporarily add a RED RECTANGLE with 'BACKEND'")
        print("text to the top-left corner of the camera image.")
        print()
        print("Steps:")
        print("  1. This script adds the marker to camera.py")
        print("  2. Backend auto-reloads (or restart manually)")
        print("  3. Open http://localhost:3000")
        print("  4. Click START")
        print("  5. Verify you see the RED RECTANGLE in browser")
        print("  6. Run: python test_backend_marker.py remove")
        print()
        
        response = input("Add test marker? (y/n): ")
        if response.lower() == 'y':
            success = add_test_marker()
        else:
            print("Cancelled")
            success = False
    
    exit(0 if success else 1)
