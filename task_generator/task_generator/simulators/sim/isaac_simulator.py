import itertools
import os
import random
import time
import typing

import arena_people_msgs.msg
import arena_simulation_setup.entities.robot
import attrs
import numpy as np
import rclpy
import rclpy.client
from isaacsim_msgs.msg import NavPed, Person
from isaacsim_msgs.srv import (
    DeletePrim,
    GetPrimAttributes,
    ImportObstacles,
    ImportUsd,
    MovePed,
    MovePrim,
    Pedestrian,
    SpawnDoor,
    SpawnFloor,
    SpawnWall,
    UrdfToUsd,
)
from std_msgs.msg import String as StdString

from task_generator.shared import (
    DynamicObstacle,
    ModelType,
    Namespace,
    Obstacle,
    Robot,
)
from task_generator.simulators.sim import BaseSim


@attrs.define()
class _Service:
    type_: typing.Any
    name: str

    _client: rclpy.client.Client = attrs.field(init=False)

    @property
    def client(self) -> rclpy.client.Client:
        if self._client is None:
            raise RuntimeError(f"client for service {self.name} not initialized")
        return self._client

    @client.setter
    def client(self, value: rclpy.client.Client):
        self._client = value


class _Services(typing.NamedTuple):
    get_prim_attributes: _Service
    urdf_to_usd: _Service
    import_usd: _Service
    import_obstacle: _Service
    move_prim: _Service
    delete_prim: _Service
    spawn_wall: _Service
    spawn_floor: _Service
    spawn_door: _Service
    import_pedestrians: _Service
    move_pedestrians: _Service
    delete_all_pedestrians: _Service


class IsaacSimulator(BaseSim):

    _NS_OBSTACLE = Namespace('Obstacles')
    _NS_PEDESTRIAN = Namespace('Pedestrians')
    _NS_ROBOT = Namespace('Robots')
    _NS_WALL = Namespace('Walls')
    _NS_FLOOR = Namespace('Floors')
    _NS_DOOR = Namespace('Doors')

    def __init__(self, namespace):
        """Initialize IsaacSimulator

        Args:
            namespace: Namespace for the simulator
        """
        super().__init__(namespace)

        self._logger.info(f"Initializing IsaacSimulator with namespace: {namespace}")

        self._init_service_clients()
        self.wall_counter = itertools.count()
        self.floor_counter = itertools.count()
        self._spawned_doors = []
        self._logger.info("Done initializing Isaac Sim")

    def robot_spawn(self, robots):
        def impl(robot: Robot) -> bool:
            try:
                model = robot.model.get(
                    [
                        ModelType.URDF,
                        # ModelType.USD
                    ],
                    loader_args=robot.asdict(),
                )

                if model.type == ModelType.URDF:
                    robot_params = arena_simulation_setup.entities.robot.Robot(robot.model.name).model_params

                    fq_name = self._NS_ROBOT(robot.name)

                    self.services.urdf_to_usd.client.call(
                        UrdfToUsd.Request(
                            name=fq_name,
                            urdf_path=model.path,
                            robot_model=robot.model.name,
                            no_localization=False,
                            base_frame=robot_params.base_frame,
                            odom_frame=robot_params.odom_frame,
                            pose=robot.pose.to_msg(),
                            cmd_vel_topic=self.node.service_namespace(robot.name, 'cmd_vel')
                        )
                    )

                    # from isaac_utils.managers.door_manager import
                    base_frame = robot_params.base_frame
                    robot_prim_path = os.path.join("/World", fq_name, base_frame)

                    # Publish registration message so DoorManager in IsaacSim process
                    # registers the robot. This avoids cross-process direct calls.
                    try:
                        if getattr(self, '_reg_pub', None) is not None:
                            self._reg_pub.publish(StdString(data=f"robot|{robot_prim_path}"))
                            self._logger.debug(f"Published registration for robot: {robot_prim_path}")
                        else:
                            self._logger.warning('Registration publisher not available; robot not registered with IsaacSim DoorManager')
                    except Exception as e:
                        self._logger.warning(f'Failed to publish robot registration: {e}')

                    return True

                # TODO
                raise NotImplementedError(
                    f"robot model of type {model.type} can't be spawned by {self.__class__.__name__}"
                )

            except Exception as e:
                self._logger.error(repr(e))
                return False

        return tuple(map(impl, robots))

    def obstacle_spawn(self, obstacles):
        results: bool = []

        for obstacle in obstacles:
            model = obstacle.model.get([ModelType.USD])
            usd_path = os.path.abspath(model.path)
            response = self.services.import_obstacle.client.call(
                ImportObstacles.Request(
                    name=self._NS_OBSTACLE(obstacle.name),
                    usd_path=usd_path,
                    pose=obstacle.pose.to_msg(),
                )
            )
            results.append(response is not None)

        return results

    def obstacle_move(self, obstacles):
        def move_obstacle(obstacle: Obstacle) -> bool:
            return self._move_entity(self._NS_OBSTACLE(obstacle.name), obstacle.pose)
        return tuple(map(move_obstacle, obstacles))

    def pedestrian_move(self, pedestrians):
        def move_pedestrian(pedestrian: Pedestrian) -> bool:
            return self._move_entity(self._NS_PEDESTRIAN(pedestrian.name), pedestrian.pose)
        return tuple(map(move_pedestrian, pedestrians))

    def robot_move(self, robots):
        def move_robot(robot: Robot) -> bool:
            return self._move_entity(self._NS_ROBOT(robot.name), robot.pose)
        return tuple(map(move_robot, robots))

    def obstacle_delete(self, obstacles):
        return tuple(self._delete_entity(self._NS_OBSTACLE(o.name)) for o in obstacles)

    def pedestrian_delete(self, pedestrians):
        return (True,) * len(pedestrians)
        # TODO uncomment when pedestrians aren't deleted immediately
        return tuple(self._delete_entity(self._NS_PEDESTRIAN(p.name)) for p in pedestrians)

    def robot_delete(self, robots):
        return tuple(self._delete_entity(self._NS_ROBOT(r.name)) for r in robots)

    def remove_walls_doors(self):
        self._delete_entity(self._NS_WALL)
        self._delete_entity(self._NS_DOOR)
        return True

    def spawn_walls(self, walls):
        # return True
        self._logger.debug("Attempting to spawn walls")

        # self.delete_walls()
        time.sleep(0.01)
        for i, wall in enumerate(walls):
            try:
                # Split wall by any doors previously spawned on this simulator
                start = np.array([wall.start.x, wall.start.y], dtype=float)
                end = np.array([wall.end.x, wall.end.y], dtype=float)
                height = getattr(wall, 'height', 2.0)

                # collect cut parameters t in [0,1]
                cuts = [0.0, 1.0]
                spawned_doors = getattr(self, '_spawned_doors', []) or []

                # compute door ranges (t_min, t_max) along this wall for skipping
                door_ranges: list[tuple[float, float]] = []

                for door in spawned_doors:
                    try:
                        dstart = np.array([door.start.x, door.start.y], dtype=float)
                        dend = np.array([door.end.x, door.end.y], dtype=float)
                    except Exception:
                        # door may be a simple mapping; try dict-like
                        try:
                            dstart = np.array(door['start'][:2], dtype=float)
                            dend = np.array(door['end'][:2], dtype=float)
                        except Exception:
                            continue

                    def _project_param(a, b, p):
                        ab = b - a
                        denom = np.dot(ab, ab)
                        if denom <= 1e-8:
                            return 0.0
                        t = float(np.dot(p - a, ab) / denom)
                        return max(0.0, min(1.0, t))

                    t0 = _project_param(start, end, dstart)
                    t1 = _project_param(start, end, dend)

                    tmin, tmax = min(t0, t1), max(t0, t1)
                    # only consider door if it overlaps the wall at all
                    if tmax <= 0.0 or tmin >= 1.0:
                        continue
                    door_ranges.append((tmin, tmax))
                    cuts.extend([tmin, tmax])

                # sanitize and sort cuts
                cuts = sorted(set([max(0.0, min(1.0, float(c))) for c in cuts]))

                # debug log door ranges
                if door_ranges:
                    self._logger.debug(f"Wall {i}: door_ranges={door_ranges}, cuts={cuts}")

                # spawn segments between successive unique cut points, skipping door intervals
                total_len = np.linalg.norm(end - start)
                EPS = 1e-3
                DOOR_EPS = 1e-3
                seg_i = 0
                spawned_segs = 0
                for a_t, b_t in zip(cuts[:-1], cuts[1:]):
                    seg_len = (b_t - a_t) * total_len
                    if seg_len < EPS:
                        self._logger.debug(f"Wall {i}: skipping tiny segment [{a_t:.4f},{b_t:.4f}] len={seg_len}")
                        continue

                    # if this interval overlaps any door range, skip it
                    overlaps_door = False
                    for dr_min, dr_max in door_ranges:
                        if not (b_t <= dr_min + DOOR_EPS or a_t >= dr_max - DOOR_EPS):
                            overlaps_door = True
                            break

                    if overlaps_door:
                        self._logger.debug(f"Wall {i}: skipping segment [{a_t:.4f},{b_t:.4f}] because it overlaps a door range")
                        continue

                    seg_start = start + (end - start) * a_t
                    seg_end = start + (end - start) * b_t

                    self.services.spawn_wall.client.call(
                        SpawnWall.Request(
                            name=self._NS_WALL(f"wall_{next(self.wall_counter)}_seg{seg_i}"),
                            start=[float(seg_start[0]), float(seg_start[1])],
                            end=[float(seg_end[0]), float(seg_end[1])],
                            height=height,
                        )
                        )
                current_count = next(self.wall_counter)
                # print(f"wall {i+1}: {wall}")
                start = [wall.start.x, wall.start.y]
                end = [wall.end.x, wall.end.y]
                type_ = wall.type_ 
                z_offset = wall.z_offset
                wall_fill_assets = wall.assets[0]
                wall_tile_assets = wall.assets[1]
                if wall.material != '':
                    wall_material = wall.material
                else:
                    wall_material = wall.assets[2]
                # print(wall_tile_assets)
                future = self.services.spawn_wall.client.call(
                    SpawnWall.Request(
                        name=f"wall_{current_count}/wall_{current_count}",
                        start=start,
                        end=end,
                        height=wall.height,
                        width=wall.width,
                        material=wall_material,
                        z_offset=z_offset
                    )
                )
                for i, wall_tile_asset in enumerate(wall_tile_assets):
                    model = wall_tile_asset.model.get([ModelType.USD])
                    usd_path = os.path.abspath(model.path)
                    response = self.services.import_obstacle.client.call(
                        ImportObstacles.Request(
                            name=f"wall_{current_count}/{wall_tile_asset.name}",
                            usd_path=usd_path,
                            pose=wall_tile_asset.pose.to_msg(),
                            type = wall_tile_asset.type_,
                        )
                    )
                for i, wall_fill_asset in enumerate(wall_fill_assets):
                    asset_start = [wall_fill_asset.start.x, wall_fill_asset.start.y]
                    asset_end = [wall_fill_asset.end.x, wall_fill_asset.end.y]
                    asset_z_offset = wall_fill_asset.z_offset
                    asset_height = wall_fill_asset.height
                    asset_width = wall_fill_asset.width
                    asset_material = wall_fill_asset.material
                    future = self.services.spawn_wall.client.call(
                    SpawnWall.Request(
                        name=f"wall_{current_count}/wall_{current_count}_asset_{i}",
                        start=asset_start,
                        end=asset_end,
                        height=asset_height,
                        width=asset_width,
                        material=asset_material,
                        z_offset=asset_z_offset,
                    )
                    )
                seg_i += 1
                spawned_segs += 1

                self._logger.info(f"Successfully spawned wall {i+1} as {spawned_segs} segment(s)")

            except Exception as e:
                self._logger.error(str(e))
                raise  # Re-raise exception after logging

        self._logger.info("All walls spawned successfully.")
        return True
        # time.sleep(0.01)

    def spawn_floors(self, floors) -> bool:
        self._logger.info("Attempting to spawn floors")
        for floor in floors:
            try:
                pos = [floor.pos.x, floor.pos.y]
                i = next(self.floor_counter)
                self.services.spawn_floor.client.call(
                    SpawnFloor.Request(
                        name=self._NS_FLOOR(f"floor_{i}"),
                        x_length=floor.x_length,
                        y_length=floor.y_length,
                        pos=pos,
                        material=floor.mat,
                    )
                )
                
                self._logger.info(f"Successfully spawned floor {i+1}")
            
            except Exception as e:
                self._logger.error(str(e))
                return False
        return True

    def spawn_doors(self, doors) -> bool:
        # cache doors so spawn_walls can split using door locations
        self._spawned_doors = doors

        for door in doors:
            self.services.spawn_door.client.call(
                SpawnDoor.Request(
                    name=self._NS_DOOR(door.name),
                    start=[door.start.x, door.start.y],
                    end=[door.end.x, door.end.y],
                    height=door.height,
                    material=door.material,
                    kind=door.kind,
                )
            )
        self._logger.info("All doors spawned successfully.")
        return True

    # TODO: update
    def before_reset_task(self):
        self._delete_all_pedestrians(self._NS_PEDESTRIAN)
        time.sleep(0.5)
        return True

    # TODO: update
    def after_reset_task(self):
        return True

    def pedestrian_spawn(self, pedestrians):

        results: list[bool] = []

        # TODO implement externally managed pedestrians
        for pedestrian in pedestrians:
            model_name = random.choice(
                [
                    # "F_Business_02",
                    # "F_Medical_01",
                    # "M_Medical_01",
                    # "biped_demo",
                    # "female_adult_police_01_new",
                    # "female_adult_police_02",
                    # "female_adult_police_03_new",
                    # "male_adult_construction_01_new",
                    # "male_adult_construction_03",
                    # "male_adult_construction_05_new",
                    # "male_adult_police_04",
                    "original_female_adult_business_02",
                    "original_female_adult_medical_01",
                    "original_female_adult_police_01",
                    "original_female_adult_police_02",
                    "original_female_adult_police_03",
                    "original_male_adult_construction_01",
                    "original_male_adult_construction_02",
                    "original_male_adult_construction_03",
                    "original_male_adult_construction_05",
                    "original_male_adult_medical_01",
                    "original_male_adult_police_04",
                ]
            )
            result = self.services.import_pedestrians.client.call(
                Pedestrian.Request(
                    people=[
                        Person(
                            stage_prefix=self._NS_PEDESTRIAN(pedestrian.name),
                            character_name=model_name,
                            initial_pose=[
                                pedestrian.pose.position.x,
                                pedestrian.pose.position.y,
                                0.0,
                            ],
                            orientation=pedestrian.pose.orientation.to_yaw(),
                            controller_stats=False,
                        )
                    ]
                )
            )
            if result is not None:
                self.ped_dict[pedestrian.name] = model_name
            results.append(result is not None)

        self.pedestrian_update(
            arena_people_msgs.msg.Pedestrians(pedestrians=[
                arena_people_msgs.msg.Pedestrian(
                    name=ped.name,
                    pose=ped.pose.to_msg(),
                )
                for ped
                in pedestrians
            ])
        )
        return True

    def pedestrian_update(self, pedestrians):
        req = MovePed.Request()

        def impl(ped: DynamicObstacle) -> bool:
            name = ped.name
            if not name in self.ped_dict:
                self._logger.warning(f"Pedestrian {name} not found in ped_dict")
                return False

            nav_ped = NavPed()
            nav_ped.path = (
                self._NS_PEDESTRIAN(name, "ManRoot", self.ped_dict[name].replace("original_", ""))
            )
            nav_ped.goal_pose = [ped.pose.position.x, ped.pose.position.y, 0.0]
            nav_ped.velocity = np.linalg.norm([ped.twist.linear.x, ped.twist.linear.y])
            req.nav_list.append(nav_ped)
            return True

        results = tuple(map(impl, pedestrians.pedestrians))

        self.services.move_pedestrians.client.call(req)
        return results

    def _init_service_clients(self):
        """
        Initialize all ROS 2 service clients and wait for their availability.
        """
        self._logger.info("Initializing service clients...")

        # Define services with their corresponding client attributes
        self.services = _Services(
            urdf_to_usd=_Service(type_=UrdfToUsd, name="isaac/urdf_to_usd"),
            import_usd=_Service(type_=ImportUsd, name="isaac/import_usd"),
            delete_prim=_Service(type_=DeletePrim, name="isaac/delete_prim"),
            get_prim_attributes=_Service(type_=GetPrimAttributes, name="isaac/get_prim_attributes"),
            move_prim=_Service(type_=MovePrim, name="isaac/move_prim"),
            spawn_wall=_Service(type_=SpawnWall, name="isaac/spawn_wall"),
            spawn_floor=_Service(type_=SpawnFloor, name='isaac/spawn_floor'),
            spawn_door=_Service(type_=SpawnDoor, name="isaac/spawn_door"),
            import_obstacle=_Service(type_=ImportObstacles, name="isaac/import_obstacle"),
            import_pedestrians=_Service(type_=Pedestrian, name="isaac/spawn_pedestrian"),
            move_pedestrians=_Service(type_=MovePed, name="isaac/move_pedestrians"),
            delete_all_pedestrians=_Service(type_=DeletePrim, name="isaac/delete_all_pedestrians"),
        )

        for service in self.services:
            service.client = self.node.create_client(service.type_, service.name)
            self._logger.debug(f'Waiting for service "{service.name}"...')

            timeout_sec = 10.0
            while not service.client.wait_for_service(timeout_sec=timeout_sec):
                self._logger.warning(
                    f'Service "{service.name}" not available after waiting {timeout_sec}s'
                )

            self._logger.debug(f'Service "{service.name}" is now available.')

        self.ped_dict = {}

        # Publisher for external registration messages so IsaacSim's DoorManager
        # can be informed about spawned entities in the IsaacSim process.
        try:
            self._reg_pub = self.node.create_publisher(StdString, '/isaac/register_entity', 10)
            self._logger.info('Created /isaac/register_entity publisher')
        except Exception as e:
            self._reg_pub = None
            self._logger.warning(f'Failed to create registration publisher: {e}')
        self._logger.info("All service clients initialized and available.")

    def _delete_entity(self, name: str) -> bool:
        self._logger.debug(f"Attempting to delete prim {name}")

        self.services.delete_prim.client.call(
            DeletePrim.Request(
                name=name
            )
        )

        return True

    def _delete_all_pedestrians(self, prim_path):
        self._logger.info(f"Attempting to delete prim named {prim_path}")

        response = self.services.delete_all_pedestrians.client.call(
            DeletePrim.Request(name=prim_path)
        )

        return True

    def _move_entity(self, name, pose):
        self._logger.debug(f"Attempting to move entity: {name}")
        self._logger.debug(f"position: {pose.position.x,pose.position.y}")
        self._logger.debug(f"orientation: {pose.orientation}")

        if name in self.ped_dict:
            name = os.path.join('pedestrians', name)

        response = self.services.move_prim.client.call(
            MovePrim.Request(
                name=name,
                pose=pose.to_msg(),
            )
        )
        if response is None:
            return False
        return True
