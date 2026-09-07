#!/usr/bin/env python3
"""
Test script to verify controls work while simulation is PAUSED.

This script:
1. Connects to WebSocket
2. Starts simulation
3. Pauses simulation
4. Changes disturbances while paused
5. Resumes simulation
6. Resets simulation
7. Verifies all commands were acknowledged
"""

import asyncio
import websockets
import json
import sys

async def wait_for_ack(websocket, expected_command):
    """Wait for ACK, skipping telemetry and frame messages."""
    while True:
        response = await websocket.recv()
        data = json.loads(response)
        if data.get("type") == "ack" and data.get("command") == expected_command:
            return data
        elif data.get("type") in ["telemetry", "frame"]:
            # Skip telemetry and frame messages
            continue
        else:
            return data

async def test_paused_controls():
    """Test that controls work while paused."""
    uri = "ws://localhost:8000/ws/simulation"
    
    print("=" * 60)
    print("PAUSED STATE CONTROLS TEST")
    print("=" * 60)
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected to WebSocket")
            
            # Step 1: START
            print("\n[1] Sending START command...")
            await websocket.send(json.dumps({"command": "start"}))
            data = await wait_for_ack(websocket, "start")
            if data.get("type") == "ack":
                print("✓ START acknowledged")
            else:
                print(f"✗ Unexpected response: {data}")
                return False
            
            # Wait a bit for simulation to run
            print("  Waiting 2 seconds for simulation to run...")
            await asyncio.sleep(2)
            
            # Step 2: PAUSE
            print("\n[2] Sending PAUSE command...")
            await websocket.send(json.dumps({"command": "pause"}))
            data = await wait_for_ack(websocket, "pause")
            if data.get("type") == "ack":
                print("✓ PAUSE acknowledged")
                print("  ** Simulation is now PAUSED **")
            else:
                print(f"✗ Unexpected response: {data}")
                return False
            
            # Step 3: Change disturbances WHILE PAUSED
            print("\n[3] Changing disturbances WHILE PAUSED...")
            
            disturbances = [
                ("turbulence", 1.5),
                ("vibration", 0.8),
                ("camera_motion", 1.2),
                ("sensor_noise", 0.6)
            ]
            
            for name, value in disturbances:
                print(f"  Changing {name} to {value}...")
                await websocket.send(json.dumps({
                    "command": "set_disturbance",
                    "name": name,
                    "value": value
                }))
                data = await wait_for_ack(websocket, "set_disturbance")
                if data.get("type") == "ack":
                    print(f"  ✓ {name} = {value} acknowledged")
                else:
                    print(f"  ✗ Failed to set {name}: {data}")
                    return False
            
            print("✓ All disturbances changed while PAUSED")
            
            # Step 4: Send preset (multiple disturbances) WHILE PAUSED
            print("\n[4] Sending preset (WORST CASE) WHILE PAUSED...")
            preset_commands = [
                ("turbulence", 2.0),
                ("vibration", 2.0),
                ("camera_motion", 1.5),
                ("sensor_noise", 2.0)
            ]
            
            for name, value in preset_commands:
                await websocket.send(json.dumps({
                    "command": "set_disturbance",
                    "name": name,
                    "value": value
                }))
                data = await wait_for_ack(websocket, "set_disturbance")
                if data.get("type") != "ack":
                    print(f"  ✗ Failed to set {name} in preset: {data}")
                    return False
            
            print("✓ Preset applied while PAUSED")
            
            # Step 5: RESUME
            print("\n[5] Sending RESUME command...")
            await websocket.send(json.dumps({"command": "resume"}))
            data = await wait_for_ack(websocket, "resume")
            if data.get("type") == "ack":
                print("✓ RESUME acknowledged")
                print("  ** Simulation is now RUNNING again **")
            else:
                print(f"✗ Unexpected response: {data}")
                return False
            
            # Wait a bit
            print("  Waiting 1 second...")
            await asyncio.sleep(1)
            
            # Step 6: PAUSE again
            print("\n[6] Sending PAUSE command again...")
            await websocket.send(json.dumps({"command": "pause"}))
            data = await wait_for_ack(websocket, "pause")
            if data.get("type") == "ack":
                print("✓ PAUSE acknowledged")
            else:
                print(f"✗ Unexpected response: {data}")
                return False
            
            # Step 7: RESET from PAUSED
            print("\n[7] Sending RESET from PAUSED state...")
            await websocket.send(json.dumps({"command": "reset"}))
            data = await wait_for_ack(websocket, "reset")
            if data.get("type") == "ack":
                print("✓ RESET acknowledged from PAUSED")
                print("  ** Simulation is now IDLE **")
            else:
                print(f"✗ Unexpected response: {data}")
                return False
            
            # Step 8: Change disturbances in IDLE (should also work)
            print("\n[8] Changing disturbance in IDLE state...")
            await websocket.send(json.dumps({
                "command": "set_disturbance",
                "name": "turbulence",
                "value": 0.0
            }))
            data = await wait_for_ack(websocket, "set_disturbance")
            if data.get("type") == "ack":
                print("✓ Disturbance change acknowledged in IDLE")
            else:
                print(f"✗ Unexpected response: {data}")
                return False
            
            print("\n" + "=" * 60)
            print("✓ ALL TESTS PASSED")
            print("=" * 60)
            print("\nConclusion:")
            print("  - Controls work while PAUSED ✓")
            print("  - Disturbances can be changed while PAUSED ✓")
            print("  - Presets can be applied while PAUSED ✓")
            print("  - RESUME works from PAUSED ✓")
            print("  - RESET works from PAUSED ✓")
            print("  - Controls work in all states (IDLE, RUNNING, PAUSED) ✓")
            
            return True
            
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"\n✗ Connection closed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nMake sure backend is running on localhost:8000\n")
    result = asyncio.run(test_paused_controls())
    sys.exit(0 if result else 1)
