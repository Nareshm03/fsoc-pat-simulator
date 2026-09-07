"""
Real WebSocket test to verify PAT state machine integration.
"""
import asyncio
import json
import websockets


async def test_pat_states():
    uri = "ws://localhost:8000/ws/simulation"
    
    print("=" * 70)
    print("PAT State Machine WebSocket Test")
    print("=" * 70)
    print()
    
    print("Connecting to WebSocket...")
    async with websockets.connect(uri) as websocket:
        print("✓ Connected\n")
        
        # Send RESET
        print("Sending RESET...")
        await websocket.send(json.dumps({"command": "reset"}))
        response = await websocket.recv()
        print(f"✓ {response}\n")
        
        # Send START
        print("Sending START...")
        await websocket.send(json.dumps({"command": "start"}))
        response = await websocket.recv()
        print(f"✓ {response}\n")
        
        # Monitor telemetry
        print("Monitoring PAT state transitions...")
        print("-" * 70)
        
        states_seen = set()
        locked_count = 0
        start_time = asyncio.get_event_loop().time()
        lock_time = None
        
        try:
            message_count = 0
            while message_count < 200 and (asyncio.get_event_loop().time() - start_time) < 8:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    data = json.loads(message)
                    message_count += 1
                    
                    if data.get("type") == "telemetry":
                        pat_state = data.get("pat", {}).get("state", "UNKNOWN")
                        error_px = data.get("error", {}).get("pixel", 0)
                        detected = data.get("detection", {}).get("detected", False)
                        sim_time = data.get("simulation_time", 0)
                        
                        states_seen.add(pat_state)
                        
                        if pat_state == "LOCKED":
                            locked_count += 1
                            if lock_time is None:
                                lock_time = sim_time
                                print(f"   t={sim_time:5.2f}s | ★ LOCKED achieved! (error={error_px:.1f}px)")
                        
                        # Print periodic updates
                        if message_count % 20 == 1 or pat_state == "LOCKED":
                            print(f"   t={sim_time:5.2f}s | State={pat_state:12s} | Det={detected} | Err={error_px:5.1f}px")
                
                except asyncio.TimeoutError:
                    pass
        
        except KeyboardInterrupt:
            print("\n⚠ Interrupted")
        
        print("-" * 70)
        print()
        
        # Send PAUSE
        print("Sending PAUSE...")
        await websocket.send(json.dumps({"command": "pause"}))
        print("✓ Paused\n")
        
        # Results
        print("=" * 70)
        print("RESULTS:")
        print("=" * 70)
        print(f"Messages received:      {message_count}")
        print(f"States observed:        {', '.join(sorted(states_seen))}")
        print(f"LOCKED achieved:        {'YES ✓' if 'LOCKED' in states_seen else 'NO ✗'}")
        if lock_time is not None:
            print(f"Lock acquisition time:  {lock_time:.2f}s")
            print(f"LOCKED frame count:     {locked_count}")
        print()
        
        # Verify
        expected_states = {'SEARCH', 'ACQUIRE', 'FINE_TRACK', 'LOCKED'}
        if states_seen >= expected_states:
            print("✓✓✓ ALL PAT STATES OBSERVED - TEST PASSED ✓✓✓")
            return True
        else:
            missing = expected_states - states_seen
            print(f"✗✗✗ Missing states: {missing} ✗✗✗")
            return False


if __name__ == "__main__":
    success = asyncio.run(test_pat_states())
    exit(0 if success else 1)
