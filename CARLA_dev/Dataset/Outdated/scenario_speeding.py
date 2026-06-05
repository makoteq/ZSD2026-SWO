import carla
import time
import random

LANE_A_Y = -170.0
LANE_B_Y = -174.0
START_X = 32.0
SPAWN_Z = 0.8
YAW = 0.0


def destroy_all_vehicles(world):
    for a in world.get_actors().filter('vehicle.*'):
        if a.is_alive:
            a.destroy()


def spawn_traffic(world, blueprint_library):
    destroy_all_vehicles(world)
    v1_model = blueprint_library.find('vehicle.dodge.charger_2020')
    v2_model = blueprint_library.find('vehicle.lincoln.mkz_2020')

    lanes = [LANE_A_Y, LANE_B_Y]
    random.shuffle(lanes)
    y_normal, y_speeder = lanes[0], lanes[1]

    t1 = carla.Transform(carla.Location(x=START_X + 20.0, y=y_normal, z=SPAWN_Z), carla.Rotation(yaw=YAW))
    t2 = carla.Transform(carla.Location(x=START_X, y=y_speeder, z=SPAWN_Z), carla.Rotation(yaw=YAW))

    v1 = world.try_spawn_actor(v1_model, t1)
    if v1 is None:
        return [], None, None

    v2 = world.try_spawn_actor(v2_model, t2)
    if v2 is None:
        v1.destroy()
        return [], None, None

    return [v1, v2], v1, v2


def run(world, blueprint_library):
    crossing_line = 180.0
    normal_speed = random.uniform(40.0, 50.0) / 3.6
    speeder_initial_speed = normal_speed
    speeder_max_speed = random.uniform(90.0, 140.0) / 3.6
    boost_trigger_x = random.uniform(60.0, 130.0)

    vehicles, v1, v2 = spawn_traffic(world, blueprint_library)
    if not v1 or not v2:
        print("[SPAWN ERROR] speeding")
        return

    try:
        scenario_active = True
        while scenario_active:
            if not v1.is_alive or not v2.is_alive:
                break

            loc1 = v1.get_location()
            loc2 = v2.get_location()

            v1.set_target_velocity(carla.Vector3D(x=normal_speed, y=0.0, z=0.0))
            v2.set_target_velocity(carla.Vector3D(
                x=speeder_max_speed if loc2.x > boost_trigger_x else speeder_initial_speed, y=0.0, z=0.0
            ))

            if loc1.x > crossing_line and loc2.x > crossing_line:
                scenario_active = False

            time.sleep(0.05)
    finally:
        destroy_all_vehicles(world)