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
            try:
                a.destroy()
            except Exception:
                pass


def get_random_car_blueprint(blueprint_library):
    # tylko osobowe, 4 koła
    blueprints = [
        bp for bp in blueprint_library.filter('vehicle.*')
        if int(bp.get_attribute('number_of_wheels')) == 4
    ]
    return random.choice(blueprints)


def run(world, blueprint_library, duration_sec=60.0):
    crossing_line = 180.0

    client = carla.Client('localhost', 2000)
    tm = client.get_trafficmanager(8000)
    tm_port = tm.get_port()

    tm.global_percentage_speed_difference(10.0)
    tm.set_global_distance_to_leading_vehicle(3.0)

    print(f"\nRozpoczynam scenariusz: Normalny ruch uliczny ({duration_sec:.0f}s).")

    destroy_all_vehicles(world)
    active_vehicles = []

    start_time = time.time()
    last_spawn_time = start_time
    next_spawn_delay = random.uniform(1.0, 2.0)

    try:
        while (time.time() - start_time) < duration_sec:
            current_time = time.time()

            if current_time - last_spawn_time > next_spawn_delay:
                spawn_y = random.choice([LANE_A_Y, LANE_B_Y])
                t = carla.Transform(
                    carla.Location(x=START_X, y=spawn_y, z=SPAWN_Z),
                    carla.Rotation(yaw=YAW)
                )
                bp = get_random_car_blueprint(blueprint_library)

                vehicle = world.try_spawn_actor(bp, t)
                if vehicle is not None:
                    vehicle.set_autopilot(True, tm_port)
                    active_vehicles.append(vehicle)
                    next_spawn_delay = random.uniform(0.8, 2.0)
                    last_spawn_time = current_time

            to_remove = []
            for v in active_vehicles:
                if not v.is_alive:
                    to_remove.append(v)
                    continue

                if v.get_location().x > crossing_line:
                    try:
                        v.set_autopilot(False, tm_port)
                    except Exception:
                        pass
                    to_remove.append(v)

            for v in to_remove:
                if v.is_alive:
                    try:
                        v.destroy()
                    except Exception:
                        pass
                if v in active_vehicles:
                    active_vehicles.remove(v)

            time.sleep(0.05)

    finally:
        for v in active_vehicles:
            if v.is_alive:
                try:
                    v.set_autopilot(False, tm_port)
                except Exception:
                    pass
                try:
                    v.destroy()
                except Exception:
                    pass
        destroy_all_vehicles(world)