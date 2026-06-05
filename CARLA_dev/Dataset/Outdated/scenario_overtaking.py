import carla
import time
import random
import math

LANE_RIGHT_Y = -170.0
LANE_LEFT_Y = -174.0
START_X = 32.0
SPAWN_Z = 0.8
YAW = 0.0


def destroy_all_vehicles(world):
    for a in world.get_actors().filter('vehicle.*'):
        if a.is_alive:
            a.destroy()


def run(world, blueprint_library):
    destroy_all_vehicles(world)

    v1_model = blueprint_library.find('vehicle.dodge.charger_2020')
    v2_model = blueprint_library.find('vehicle.lincoln.mkz_2020')

    t1 = carla.Transform(carla.Location(x=START_X + 30.0, y=LANE_RIGHT_Y, z=SPAWN_Z), carla.Rotation(yaw=YAW))
    t2 = carla.Transform(carla.Location(x=START_X, y=LANE_RIGHT_Y, z=SPAWN_Z), carla.Rotation(yaw=YAW))

    v1 = world.try_spawn_actor(v1_model, t1)
    v2 = world.try_spawn_actor(v2_model, t2)

    if v1 is None or v2 is None:
        if v1 is not None and v1.is_alive:
            v1.destroy()
        if v2 is not None and v2.is_alive:
            v2.destroy()
        print("[SPAWN ERROR] overtaking")
        return

    crossing_line = 180.0
    vehicle_length = 5.0
    lane_width = abs(LANE_RIGHT_Y - LANE_LEFT_Y)

    normal_speed = random.uniform(40.0, 50.0) / 3.6
    speeder_speed = random.uniform(90.0, 130.0) / 3.6
    lateral_speed = random.uniform(4.0, 6.0)

    relative_speed = speeder_speed - normal_speed
    time_to_change_lane = lane_width / lateral_speed
    closing_distance = relative_speed * time_to_change_lane

    pull_out_clearance = random.uniform(2.0, 4.0)
    pull_out_trigger = vehicle_length + closing_distance + pull_out_clearance

    pull_in_clearance = random.uniform(3.0, 6.0)
    pull_in_trigger = vehicle_length + pull_in_clearance

    try:
        state = 0
        scenario_active = True

        while scenario_active:
            if not v1.is_alive or not v2.is_alive:
                break

            loc1 = v1.get_location()
            loc2 = v2.get_location()

            v1.set_target_velocity(carla.Vector3D(x=normal_speed, y=0.0, z=0.0))

            if state == 0:
                v2.set_target_velocity(carla.Vector3D(x=speeder_speed, y=0.0, z=0.0))
                if loc1.x - loc2.x < pull_out_trigger:
                    state = 1

            elif state == 1:
                vy = -lateral_speed
                yaw = math.degrees(math.atan2(vy, speeder_speed))
                v2.set_target_velocity(carla.Vector3D(x=speeder_speed, y=vy, z=0.0))
                t2m = v2.get_transform()
                t2m.rotation.yaw = yaw
                v2.set_transform(t2m)

                if loc2.y <= LANE_LEFT_Y:
                    state = 2

            elif state == 2:
                v2.set_target_velocity(carla.Vector3D(x=speeder_speed, y=0.0, z=0.0))
                t2m = v2.get_transform()
                t2m.rotation.yaw = 0.0
                t2m.location.y = LANE_LEFT_Y
                v2.set_transform(t2m)

                if loc2.x - loc1.x > pull_in_trigger:
                    state = 3

            elif state == 3:
                vy = lateral_speed
                yaw = math.degrees(math.atan2(vy, speeder_speed))
                v2.set_target_velocity(carla.Vector3D(x=speeder_speed, y=vy, z=0.0))
                t2m = v2.get_transform()
                t2m.rotation.yaw = yaw
                v2.set_transform(t2m)

                if loc2.y >= LANE_RIGHT_Y:
                    state = 4

            elif state == 4:
                v2.set_target_velocity(carla.Vector3D(x=speeder_speed, y=0.0, z=0.0))
                t2m = v2.get_transform()
                t2m.rotation.yaw = 0.0
                t2m.location.y = LANE_RIGHT_Y
                v2.set_transform(t2m)

            if loc1.x > crossing_line:
                scenario_active = False

            time.sleep(0.05)
    finally:
        destroy_all_vehicles(world)