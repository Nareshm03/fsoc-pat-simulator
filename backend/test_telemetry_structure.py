"""
Test script to verify telemetry structure sent by backend.
"""
import asyncio
import json
import websockets

async def test_telemetry():
    uri = "ws://localhost:8000/ws/simulation"
    
    print("Connecting to WebSocket...")
    async with websockets.connect(uri) as websocket:
        print("✓ Connected\n")
        
        # Send START command
        await websocket.send(json.dumps({"command": "start"}))
        print("✓ Sent START command\n")
        
        # Receive messages
        message_count = 0
        frame_count = 0
        telemetry_count = 0
        
        try:
            while message_count < 15:  # Receive 15 messages
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                message_count += 1
                
                if data.get("type") == "telemetry":
                    telemetry_count += 1
                    if telemetry_count == 1:
                        print("=" * 60)
                        print("FIRST TELEMETRY MESSAGE STRUCTURE:")
                        print("=" * 60)
                        print(json.dumps(data, indent=2))
                        print("\n" + "=" * 60)
                        print("EXTRACTED VALUES FOR FRONTEND:")
                        print("=" * 60)
                        print(f"PAT State:    {data.get('pat', {}).get('state', 'N/A')}")
                        print(f"Beacon X:     {data.get('detection', {}).get('x', 'N/A')}")
                        print(f"Beacon Y:     {data.get('detection', {}).get('y', 'N/A')}")
                        print(f"Error (px):   {data.get('error', {}).get('pixel', 'N/A')}")
                        print(f"Sim FPS:      {data.get('performance', {}).get('simulation_fps', 'N/A')}")
                        print(f"Proc FPS:     {data.get('performance', {}).get('processing_fps', 'N/A')}")
                        print(f"Azimuth:      {data.get('gimbal', {}).get('azimuth', 'N/A')}")
                        print(f"Elevation:    {data.get('gimbal', {}).get('elevation', 'N/A')}")
                        print(f"Confidence:   {data.get('detection', {}).get('confidence', 'N/A')}")
                        print(f"Link Status:  {data.get('fsoc', {}).get('status', 'N/A')}")
                        print(f"Detected:     {data.get('detection', {}).get('detected', 'N/A')}")
                        print("=" * 60)
                        print()
                    elif telemetry_count <= 5:
                        print(f"Telemetry #{telemetry_count}: "
                              f"state={data.get('pat', {}).get('state')}, "
                              f"beacon=({data.get('detection', {}).get('x', 0):.1f}, "
                              f"{data.get('detection', {}).get('y', 0):.1f}), "
                               f"error={data.get('error', {}).get('pixel', 0):.1f}px, "
                               f"fps={data.get('performance', {}).get('processing_fps', 0):.1f}")
                
                elif data.get("type") == "frame":
                    frame_count += 1
                    print(f"Frame #{frame_count}: frame_number={data.get('frame')}, "
                          f"time={data.get('simulation_time'):.2f}s, "
                          f"image_size={len(data.get('image', ''))} chars")
        
        except asyncio.TimeoutError:
            print("\n⚠ Timeout - no more messages")
        
        # Send PAUSE
        await websocket.send(json.dumps({"command": "pause"}))
        print("\n✓ Sent PAUSE command")
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        print(f"Total messages:    {message_count}")
        print(f"Telemetry msgs:    {telemetry_count}")
        print(f"Frame msgs:        {frame_count}")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_telemetry())
