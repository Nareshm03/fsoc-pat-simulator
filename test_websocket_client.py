"""
Simple WebSocket client test to verify the backend is working
"""
import asyncio
import json
import websockets


async def test_websocket():
    uri = "ws://localhost:8000/ws/simulation"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✓ Connected to {uri}")
            
            # Send start command
            await websocket.send(json.dumps({"command": "start"}))
            print("✓ Sent START command")
            
            # Receive a few telemetry messages
            for i in range(5):
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get("type") == "telemetry":
                    print(f"✓ Received telemetry (tick {data.get('tick', 'unknown')})")
                    print(f"   State: {data.get('state')}")
                    print(f"   FPS: {data.get('fps', 0):.1f}")
                    print(f"   Error: {data.get('tracking_error', 0):.2f} px")
                elif data.get("type") == "status":
                    print(f"✓ Status: {data.get('message')}")
                
                await asyncio.sleep(0.1)
            
            # Send pause command
            await websocket.send(json.dumps({"command": "pause"}))
            print("✓ Sent PAUSE command")
            
            # Wait for response
            message = await websocket.recv()
            data = json.loads(message)
            print(f"✓ Response: {data}")
            
            print("\n✅ WebSocket test passed!")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_websocket())
