from stable_baselines3.common.callbacks import BaseCallback
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType
import time

class InitiateNewTrainStage(BaseCallback):
    def __init__(self, node: Node, train_stages: dict, threshold_type: str, upper_threshold: float,
                 lower_threshold: float, num_envs: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.node = node
        self.train_stages = train_stages
        self.threshold_type = threshold_type
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.num_envs = num_envs
        self.TIMEOUT = 10.0
        
        # Init other needed variables
        self.curriculum_index = 0
        self.max_index = len(next(iter(train_stages.values())))
        self.parameter_clients = self._init_parameter_clients()

    def _init_parameter_clients(self):
        clients = {}
        for i in range(self.num_envs):
            node_name = f"/task_generator_node_{i}" if self.num_envs > 1 else "/task_generator_node"
            clients[node_name] = self.node.create_client(SetParameters, f"{node_name}/set_parameters")
        return clients

    def _set_parameters_batch(self, node_name: str, param_dict: dict) -> bool:
        """Set parameters for a single node - returns True on success"""
        client = self.parameter_clients.get(node_name)
        if not client:
            if self.verbose > 0:
                print(f"No parameter client found for {node_name}")
            return False

        if not client.wait_for_service(timeout_sec=self.TIMEOUT):
            if self.verbose > 0:
                print(f"Service for {node_name} unavailable after timeout")
            return False

        params = []
        for param_name, param_value in param_dict.items():
            if isinstance(param_value, list) and not param_value:
                if self.verbose > 0:
                    print(f"Skipping empty parameter {param_name} for {node_name}")
                continue

            param = Parameter()
            param.name = param_name

            if isinstance(param_value, int):
                param.value.type = ParameterType.PARAMETER_INTEGER
                param.value.integer_value = param_value
            elif isinstance(param_value, float):
                param.value.type = ParameterType.PARAMETER_DOUBLE
                param.value.double_value = param_value
            elif isinstance(param_value, str):
                param.value.type = ParameterType.PARAMETER_STRING
                param.value.string_value = param_value
            elif isinstance(param_value, bool):
                param.value.type = ParameterType.PARAMETER_BOOL
                param.value.bool_value = param_value
            elif isinstance(param_value, list) and all(isinstance(x, int) for x in param_value):
                param.value.type = ParameterType.PARAMETER_INTEGER_ARRAY
                param.value.integer_array_value = param_value
            elif isinstance(param_value, list) and all(isinstance(x, float) for x in param_value):
                param.value.type = ParameterType.PARAMETER_DOUBLE_ARRAY
                param.value.double_array_value = param_value
            elif isinstance(param_value, list) and all(isinstance(x, str) for x in param_value):
                param.value.type = ParameterType.PARAMETER_STRING_ARRAY
                param.value.string_array_value = param_value
            elif isinstance(param_value, list) and all(isinstance(x, bool) for x in param_value):
                param.value.type = ParameterType.PARAMETER_BOOL_ARRAY
                param.value.bool_array_value = param_value
            elif isinstance(param_value, list) and all(isinstance(x, bytes) for x in param_value):
                param.value.type = ParameterType.PARAMETER_BYTE_ARRAY
                param.value.byte_array_value = param_value
            else:
                if self.verbose > 0:
                    print(f"Unsupported parameter type for {param_name}: {type(param_value)}")
                return False

            params.append(param)

        request = SetParameters.Request(parameters=params)
        
        try:
            future = client.call_async(request)
            
            # Wait for the future with timeout
            start_time = time.time()
            while not future.done():
                import rclpy
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if time.time() - start_time > self.TIMEOUT:
                    if self.verbose > 0:
                        print(f"Timeout waiting for response from {node_name}")
                    return False

            response = future.result()

            if response and all(result.successful for result in response.results):
                if self.verbose > 0:
                    print(f"Parameters set successfully on {node_name}")
                return True
            else:
                if self.verbose > 0:
                    print(f"Failed to set parameters on {node_name}")
                    if response:
                        for i, result in enumerate(response.results):
                            if not result.successful:
                                print(f"  Parameter {i}: {result.reason}")
                return False

        except Exception as e:
            if self.verbose > 0:
                print(f"Exception setting parameters on {node_name}: {e}")
            return False

    def _set_parameters(self, param_dict: dict) -> bool:
        """Set parameters for all task generator nodes"""
        if not self.parameter_clients:
            if self.verbose > 0:
                print("No parameter clients available")
            return False

        success_count = 0
        for node_name in self.parameter_clients.keys():
            if self._set_parameters_batch(node_name, param_dict):
                success_count += 1
            else:
                if self.verbose > 0:
                    print(f"Failed to set parameters for {node_name}")

        success = success_count == len(self.parameter_clients)
        if self.verbose > 0:
            print(f"Parameter setting complete: {success_count}/{len(self.parameter_clients)} successful")
        return success

    def _apply_curriculum(self):
        """Apply current curriculum stage parameters"""
        try:
            param_values = {
                param_name: values[self.curriculum_index]
                for param_name, values in self.train_stages.items()
            }
            
            if self.verbose > 0:
                print(f"Applying curriculum stage {self.curriculum_index}: {param_values}")
            
            return self._set_parameters(param_values)
        except Exception as e:
            if self.verbose > 0:
                print(f"Error applying curriculum: {e}")
            return False

    def _advance_curriculum(self):
        if self.curriculum_index < self.max_index - 1:
            self.curriculum_index += 1
            if self.verbose > 0:
                print(f"Advanced to curriculum stage {self.curriculum_index}")
            self._apply_curriculum()

    def _retreat_curriculum(self):
        if self.curriculum_index > 0:
            self.curriculum_index -= 1
            if self.verbose > 0:
                print(f"Retreated to curriculum stage {self.curriculum_index}")
            self._apply_curriculum()

    def _on_step(self) -> bool:
        # Access performance data from the parent EvalCallback
        # The parent callback is available through self.parent
        if hasattr(self, 'parent') and self.parent is not None:
            eval_callback = self.parent
            
            if self.threshold_type == "rew":
                current_performance = eval_callback.best_mean_reward
            elif self.threshold_type == "succ":
                current_performance = getattr(eval_callback, 'last_success_rate', 0.0)
            else:
                return True
                
            # Check thresholds and apply curriculum changes
            if current_performance >= self.upper_threshold:
                self._advance_curriculum()
                # Reset performance tracking for next stage
                if self.threshold_type == "rew":
                    eval_callback.best_mean_reward = float('-inf')
                elif self.threshold_type == "succ":
                    eval_callback.last_success_rate = 0.0
                    
            elif current_performance <= self.lower_threshold:
                self._retreat_curriculum()
        
        return True