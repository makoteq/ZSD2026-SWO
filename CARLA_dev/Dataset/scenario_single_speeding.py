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


def run(world, blueprint_library):
    destroy_all_vehicles(world)

    v1_model = blueprint_library.find('vehicle.dodge.charger_2020')
    v2_model = blueprint_library.find('vehicle.lincoln.mkz_2020')
    y_lane = random.choice([LANE_A_Y, LANE_B_Y])

    t1 = carla.Transform(carla.Location(x=START_X, y=y_lane, z=SPAWN_Z), carla.Rotation(yaw=YAW))
    v1 = world.try_spawn_actor(v1_model, t1)

    if v1 is None:
        print("[SPAWN ERROR] single_speeding")
        return

    crossing_line = 180.0
    initial_speed = random.uniform(40.0, 60.0) / 3.6
    max_speed = random.uniform(100.0, 150.0) / 3.6
    boost_trigger_x = random.uniform(60.0, 130.0)

    try:
        scenario_active = True
        while scenario_active:
            if not v1.is_alive:
                break

            loc1 = v1.get_location()
            v1.set_target_velocity(carla.Vector3D(
                x=max_speed if loc1.x > boost_trigger_x else initial_speed, y=0.0, z=0.0
            ))

            if loc1.x > crossing_line:
                scenario_active = False

            time.sleep(0.05)
    finally:
        destroy_all_vehicles(world)