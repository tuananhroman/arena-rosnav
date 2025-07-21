#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time

from staged_train_callback import InitiateNewTrainStage

class TestNode(Node):
    def __init__(self):
        super().__init__('test_staged_callback')

class MockEvalCallback:
    """Mock evaluation callback"""
    def __init__(self):
        self.best_mean_reward = -100.0  # Start low
        self.last_success_rate = 0.1    # Start low

def test_parameter_setting():
    rclpy.init()
    
    # Create a test node
    test_node = TestNode()
    
    # Define test curriculum stages
    test_stages = {
        "task.random.static.n": [[1,1], [2,2], [3,3], [4,4]],
        "task.random.dynamic.n": [[11,11], [12,12], [13,13], [14,14]],
        "goal_tolerance_radius": [0.5, 0.4, 0.3, 0.2]
    }
    
    # Create the callback with 2 environments
    callback = InitiateNewTrainStage(
        node=test_node,
        train_stages=test_stages,
        threshold_type="rew",
        upper_threshold=50.0,
        lower_threshold=-50.0,
        num_envs=2, 
        verbose=1
    )
    
    # Create mock parent callback for _on_step testing
    mock_parent = MockEvalCallback()
    callback.parent = mock_parent
    
    print("=== Testing Staged Training Callback ===")
    print(f"Number of environments: {callback.num_envs}")
    print(f"Parameter clients: {list(callback.parameter_clients.keys())}")
    
    # Test manual curriculum advancement
    print("\n--- Testing Manual Curriculum Control ---")
    print(f"Initial stage: {callback.curriculum_index}")
    callback._apply_curriculum()
    
    time.sleep(2)  # Wait for parameters to be set
    
    # Advance to next stage
    print("Advancing curriculum...")
    callback._advance_curriculum()
    
    time.sleep(2)
    
    # Advance again
    print("Advancing curriculum again...")
    callback._advance_curriculum()
    
    time.sleep(2)
    
    # Test retreat
    print("Retreating curriculum...")
    callback._retreat_curriculum()
    
    time.sleep(2)
    
    # Test _on_step function
    print("\n--- Testing _on_step Function ---")
    
    # Test 1: Performance below lower threshold (should retreat)
    print(f"\nTest 1: Low performance (reward = -75, below lower threshold of {callback.lower_threshold})")
    callback.curriculum_index = 2  # Set to stage 2 so we can retreat
    initial_stage = callback.curriculum_index
    print(f"Current stage before _on_step: {initial_stage}")
    
    mock_parent.best_mean_reward = -75.0  # Below lower threshold
    result = callback._on_step()
    
    print(f"Current stage after _on_step: {callback.curriculum_index}")
    print(f"_on_step returned: {result}")
    
    if callback.curriculum_index < initial_stage:
        print("✓ Successfully retreated due to low performance")
    else:
        print("✗ Failed to retreat on low performance")
    
    time.sleep(2)
    
    # Test 2: Performance above upper threshold (should advance)
    print(f"\nTest 2: High performance (reward = 75, above upper threshold of {callback.upper_threshold})")
    initial_stage = callback.curriculum_index
    print(f"Current stage before _on_step: {initial_stage}")
    
    mock_parent.best_mean_reward = 75.0  # Above upper threshold
    result = callback._on_step()
    
    print(f"Current stage after _on_step: {callback.curriculum_index}")
    print(f"Reward after reset: {mock_parent.best_mean_reward}")  # Should be reset
    
    if callback.curriculum_index > initial_stage:
        print("✓ Successfully advanced due to high performance")
    else:
        print("✗ Failed to advance on high performance")
    
    time.sleep(2)
    
    # Test 3: Performance in middle range (should not change)
    print(f"\nTest 3: Medium performance (reward = 0, between thresholds)")
    initial_stage = callback.curriculum_index
    print(f"Current stage before _on_step: {initial_stage}")
    
    mock_parent.best_mean_reward = 0.0  # Between thresholds
    result = callback._on_step()
    
    print(f"Current stage after _on_step: {callback.curriculum_index}")
    
    if callback.curriculum_index == initial_stage:
        print("✓ Correctly maintained stage for medium performance")
    else:
        print("✗ Incorrectly changed stage for medium performance")
    
    time.sleep(2)
    
    # Test 4: Success rate threshold testing
    print(f"\nTest 4: Testing success rate thresholds")
    callback.threshold_type = "succ"
    callback.upper_threshold = 0.8
    callback.lower_threshold = 0.2
    
    initial_stage = callback.curriculum_index
    print(f"Current stage before _on_step: {initial_stage}")
    print(f"Setting success rate to 0.9 (above upper threshold of 0.8)")
    
    mock_parent.last_success_rate = 0.9  # High success rate
    result = callback._on_step()
    
    print(f"Current stage after _on_step: {callback.curriculum_index}")
    print(f"Success rate after reset: {mock_parent.last_success_rate}")
    
    if callback.curriculum_index > initial_stage:
        print("✓ Successfully advanced due to high success rate")
    else:
        print("✗ Failed to advance on high success rate")
    
    time.sleep(2)
    
    # Test 5: Boundary conditions
    print(f"\nTest 5: Testing boundary conditions")
    
    # Test minimum boundary
    callback.curriculum_index = 0
    callback.threshold_type = "rew"
    callback.upper_threshold = 50.0
    callback.lower_threshold = -50.0
    mock_parent.best_mean_reward = -75.0  # Should try to retreat
    
    print(f"At minimum stage (0), trying to retreat...")
    result = callback._on_step()
    
    if callback.curriculum_index == 0:
        print("✓ Correctly stayed at minimum stage")
    else:
        print("✗ Incorrectly moved from minimum stage")
    
    # Test maximum boundary
    callback.curriculum_index = callback.max_index - 1  # Last stage
    mock_parent.best_mean_reward = 75.0  # Should try to advance
    
    print(f"At maximum stage ({callback.max_index - 1}), trying to advance...")
    initial_stage = callback.curriculum_index
    result = callback._on_step()
    
    if callback.curriculum_index == initial_stage:
        print("✓ Correctly stayed at maximum stage")
    else:
        print("✗ Incorrectly moved from maximum stage")
    
    print("\n=== Test Summary ===")
    print("✓ Manual curriculum control tested")
    print("✓ _on_step function tested with reward thresholds")
    print("✓ _on_step function tested with success rate thresholds")
    print("✓ Boundary conditions tested")
    print("✓ Parameter setting tested for 2 environments")
    
    # Clean up
    test_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    test_parameter_setting()