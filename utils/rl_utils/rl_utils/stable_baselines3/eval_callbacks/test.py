#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time

# Import your callback
from staged_train_callback import InitiateNewTrainStage

class TestNode(Node):
    def __init__(self):
        super().__init__('test_staged_callback')

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
    
    # Create the callback
    callback = InitiateNewTrainStage(
        node=test_node,
        train_stages=test_stages,
        threshold_type="rew",
        upper_threshold=100.0,
        lower_threshold=50.0,
        num_envs=1,  # Adjust based on your setup
        verbose=1
    )
    
    # Test manual curriculum advancement
    print("Testing manual curriculum advancement...")
    
    # Test current stage (should be stage 0)
    print(f"Current stage: {callback.curriculum_index}")
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
    
    # Clean up
    test_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    test_parameter_setting()