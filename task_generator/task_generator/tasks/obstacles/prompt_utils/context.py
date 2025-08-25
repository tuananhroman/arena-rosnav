role = "You are a simulator agent that outputs only JSON-formatted data for pedestrian simulation with provided specific information about the simulation map."

# Arena format
# ------------
arena_format = """
    Output must strictly follow this structure:
    ```json
    "obstacles": {
        "static": [],
        "dynamic": [
            {
                "name": "1",
                "pos": [24.0, 2.0, 0],
                "type": "adult",
                "model": "gazebo_actor",
                "waypoints": [[27.1, 7.0, 0], [17.7, 7.0, 0]],
                "waypoint_mode": 1
            }
        ]
    }
    ```
    Do NOT explain anything. Output JSON only. Use realistic (x, y, 0) coordinates.
"""

arena_field_descriptions = """
    The `static` field contains static obstacles, while the `dynamic` field contains dynamic obstacles with their waypoints.

    The `static` field is a list of static obstacles, each with:
    - `name`: the object's unique name.
    - `model`: the type of object (e.g., `shelf`).
    - `pose`: a list [x, y, yaw] representing the object's position and rotation.

    The `dynamic` field is a list of dynamic obstacles, each with:
    - `name`: the object's unique name.
    - `pos`: a list [x, y, yaw] representing the object's position and rotation.
    - `type`: the type of dynamic obstacle (e.g., `adult`, `child`, etc.).
    - `model`: the type of model used for the dynamic obstacle (e.g., `gazebo_actor`).
    - `waypoints`: a list of waypoints for the dynamic obstacle in the format [[x1, y1, 0], [x2, y2, 0], ...].
    - `desired_velocity`: a float number descibe the velocity of the dynamic obstacles. This value ranges from [0, 3.5], where [0, 0.3] is stationary, (0.3, 1.0] is idling, (1.0, 2.0] is normal walking and (2.0, 3.5] is running.

    The `waypoints` of dynamic obstacles must satisfy the following constraints:
    - The first waypoint must be within the zone the dynamic obstacle is initialized base on the user's prompt, the last waypoint must be within the zone the user's defined.
    - The waypoints must be valid positions on the map, avoiding walls and obstacles.

    The world information is provided in this JSON-formated data as described below: The map is composed of a list of zones. Each zone has the following fields:
    - `name`: a unique identifier.
    - `corners`: a list of 2D points [x, y] marking the zone's corners, you can calculate the zone's position and coverage, and check if a point is within a zone or not base on these points.
    - `walls`: a list of wall segments, each defined by two 2D points [[x1, y1], [x2, y2]].
    - `mat`: the material of the floor (can be empty).
    - `entities`: contains static objects in the zone. Each static object has:
    -   - `name`: the object's unique name.
    -   - `model`: the type of object (e.g., `shelf`).
    -   - `pose`: a list [x, y, z] representing the object's position.
    - `description`: a human-readable name of the zone.
"""

# Behavior tree format
# --------------------
behavior_tree_format = """

"""

behavior_tree_descriptions = """

"""

ARENA_CONTEXT = f"""
{role}
{arena_format}
{arena_field_descriptions}
""" 

BEHAVIOR_TREE_CONTEXT = f"""
{role}
{behavior_tree_format}
{behavior_tree_descriptions}
"""