"""Simple test to check if simulation runs"""
import asyncio
import json
import websockets


async def test():
    uri = "ws://localhost:8000/ws/simulation"
    
    async with websockets.connect(uri) as ws:
        print("Connected")
        
        # Send start
        await ws.send(json.dumps({"command": "start"}))
        print("Sent START")
        
        # Wait for messages
        for i in range(10):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(msg)
                print(f"Received: type={data.get('type')}, frame={data.get('frame', 'N/A')}")
            except asyncio.TimeoutError:
                print("Timeout")
                break
        
        print("Done")


if __name__ == "__main__":
    asyncio.run(test())
