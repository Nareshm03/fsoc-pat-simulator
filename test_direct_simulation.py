"""
Direct test of simulation loop without WebSocket
"""
import sys
sys.path.insert(0, 'backend')

import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

from main import SimulationManager


async def test_simulation():
    """Test simulation directly"""
    print("Creating SimulationManager...")
    manager = SimulationManager()
    
    print(f"Initial state: running={manager.running}, frame={manager.frame_number}")
    
    print("\nCalling start()...")
    await manager.start()
    
    print(f"After start(): running={manager.running}, frame={manager.frame_number}")
    print(f"Task: {manager.simulation_task}")
    print(f"Task done: {manager.simulation_task.done() if manager.simulation_task else 'No task'}")
    
    # Wait a bit for simulation to run
    print("\nWaiting 2 seconds for simulation...")
    await asyncio.sleep(2)
    
    print(f"\nAfter 2s: running={manager.running}, frame={manager.frame_number}, time={manager.simulation_time:.2f}s")
    print(f"Task done: {manager.simulation_task.done() if manager.simulation_task else 'No task'}")
    
    if manager.simulation_task and manager.simulation_task.done():
        try:
            result = manager.simulation_task.result()
            print(f"Task result: {result}")
        except Exception as e:
            print(f"Task exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Stop simulation
    print("\nStopping simulation...")
    await manager.stop()
    
    print(f"Final: frame={manager.frame_number}, time={manager.simulation_time:.2f}s")
    
    if manager.frame_number > 10:
        print("\n✅ SUCCESS: Simulation advanced!")
        return True
    else:
        print("\n❌ FAIL: Simulation did not advance")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_simulation())
    exit(0 if success else 1)
