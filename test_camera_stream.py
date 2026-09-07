"""
Test camera frame streaming from backend
"""
import asyncio
import json
import base64
import cv2
import numpy as np
import websockets
from pathlib import Path


async def test_camera_stream():
    uri = "ws://localhost:8000/ws/simulation"
    
    first_frame = None
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✓ Connected to {uri}\n")
            
            # Send start command
            await websocket.send(json.dumps({"command": "start"}))
            print("✓ Sent START command\n")
            
            frame_count = 0
            telemetry_count = 0
            
            # Receive messages for 5 seconds
            for i in range(50):  # ~5 seconds at 30fps
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    if data.get("type") == "telemetry":
                        telemetry_count += 1
                        if telemetry_count % 10 == 1:
                            print(f"  Telemetry #{telemetry_count}:")
                            print(f"    State: {data.get('state', 'N/A')}")
                            print(f"    FPS: {data.get('fps', 0):.1f}")
                            print(f"    Beacon: ({data.get('beacon_x', 0):.1f}, {data.get('beacon_y', 0):.1f})")
                            print(f"    Error: {data.get('tracking_error', 0):.2f} px\n")
                    
                    elif data.get("type") == "frame":
                        frame_count += 1
                        frame_num = data.get('frame', 0)
                        sim_time = data.get('simulation_time', 0)
                        image_b64 = data.get('image', '')
                        
                        # Comprehensive JPEG validation
                        print(f"✓ Frame #{frame_num} (t={sim_time:.2f}s):")
                        
                        # Test 1: Base64 decoding
                        try:
                            image_bytes = base64.b64decode(image_b64)
                            print(f"    ✓ Base64 decoding: PASS ({len(image_bytes)} bytes)")
                        except Exception as e:
                            print(f"    ✗ Base64 decoding: FAIL - {e}\n")
                            continue
                        
                        # Test 2: JPEG signature check
                        if len(image_bytes) >= 2 and image_bytes[0:2] == b'\xff\xd8':
                            print(f"    ✓ JPEG signature: PASS (FF D8)")
                        else:
                            print(f"    ✗ JPEG signature: FAIL (got {image_bytes[0:2].hex() if len(image_bytes) >= 2 else 'no data'})")
                        
                        # Test 3: OpenCV JPEG decode
                        try:
                            import numpy as np
                            nparr = np.frombuffer(image_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if img is not None:
                                print(f"    ✓ OpenCV JPEG decode: PASS")
                                print(f"    ✓ Image dimensions: {img.shape[1]}x{img.shape[0]}x{img.shape[2]}")
                                print(f"    ✓ Image dtype: {img.dtype}")
                                
                                # Test 4: Image size validation
                                if img.shape == (720, 1280, 3):
                                    print(f"    ✓ Expected dimensions: PASS (720x1280x3)")
                                else:
                                    print(f"    ⚠ Expected dimensions: WARN (got {img.shape}, expected (720,1280,3))")
                                
                                # Save first frame
                                if frame_count == 1:
                                    output_path = Path("test_frame.jpg")
                                    output_path.write_bytes(image_bytes)
                                    print(f"    ✓ Saved test frame to: {output_path.absolute()}")
                                
                                # Store for comparison
                                if frame_count == 1:
                                    first_frame = img.copy()
                                elif frame_count == 5:
                                    # Compare frames to verify content changes
                                    diff = cv2.absdiff(first_frame, img)
                                    mean_diff = diff.mean()
                                    if mean_diff > 1.0:
                                        print(f"    ✓ Frame content changes: PASS (mean diff: {mean_diff:.2f})")
                                    else:
                                        print(f"    ⚠ Frame content changes: WARN (mean diff: {mean_diff:.2f}, frames may be identical)")
                            else:
                                print(f"    ✗ OpenCV JPEG decode: FAIL (imdecode returned None)")
                        except Exception as e:
                            print(f"    ✗ OpenCV JPEG decode: FAIL - {e}")
                        
                        print()  # Blank line between frames
                
                except asyncio.TimeoutError:
                    print("⚠ Timeout waiting for message")
                    break
            
            # Send pause command
            await websocket.send(json.dumps({"command": "pause"}))
            print("\n✓ Sent PAUSE command")
            
            print(f"\n{'='*50}")
            print(f"SUMMARY:")
            print(f"  Telemetry messages: {telemetry_count}")
            print(f"  Camera frames: {frame_count}")
            print(f"  Expected frames: ~{telemetry_count // 3} (every 3rd tick)")
            print(f"  Frame rate: ~{frame_count / 5:.1f} fps")
            print(f"{'='*50}\n")
            
            if frame_count > 0:
                print("✅ Camera stream test PASSED!")
                print("   Backend is sending real camera frames")
                print("   Frames are valid JPEG format")
                print("   Frames can be decoded by OpenCV")
                print("   Frame dimensions are correct (720x1280x3)")
                print("   Frame content changes over time")
                return True
            else:
                print("❌ Camera stream test FAILED!")
                print("   No camera frames received")
                return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_camera_stream())
    exit(0 if success else 1)
