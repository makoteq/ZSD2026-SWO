import carla
import time
import random


def get_random_car_blueprint(blueprint_library):
    blueprints = [bp for bp in blueprint_library.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) == 4]
    return random.choice(blueprints)


def run(world, blueprint_library):
    crossing_line = 180.0

    client = carla.Client('localhost', 2000)
    tm = client.get_trafficmanager(8000)
    tm_port = tm.get_port()


    tm.global_percentage_speed_difference(10.0)
    tm.set_global_distance_to_leading_vehicle(3.0)

    print("\nRozpoczynam scenariusz: Normalny ruch uliczny (Nieskończony). Naciśnij Ctrl+C, aby wrócić.")

    active_vehicles = []
    last_spawn_time = time.time()


    next_spawn_delay = random.uniform(2.0, 4.0)

    try:
        while True:
            current_time = time.time()

            if current_time - last_spawn_time > next_spawn_delay:
                spawn_y = random.choice([1.75, 5.25])
                spawn_transform = carla.Transform(carla.Location(x=10.0, y=spawn_y, z=0.05), carla.Rotation(yaw=0.0))

                bp = get_random_car_blueprint(blueprint_library)

                vehicle = world.try_spawn_actor(bp, spawn_transform)

                if vehicle is not None:
                    vehicle.set_autopilot(True, tm_port)
                    active_vehicles.append(vehicle)
                    next_spawn_delay = random.uniform(1.5, 3.5)
                    last_spawn_time = current_time

            vehicles_to_remove = []
            for v in active_vehicles:
                if not v.is_alive:
                    vehicles_to_remove.append(v)
                    continue

                loc = v.get_location()
                if loc.x > crossing_line:
                    v.set_autopilot(False, tm_port)
                    vehicles_to_remove.append(v)

            for v in vehicles_to_remove:
                if v.is_alive:
                    v.destroy()
                if v in active_vehicles:
                    active_vehicles.remove(v)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nPrzerwano scenariusz. Czyszczenie ruchu...")
    finally:
        for v in active_vehicles:
            if v.is_alive:
                v.set_autopilot(False, tm_port)
                v.destroy()

        actors = world.get_actors().filter('vehicle.*')
        for a in actors:
            a.set_autopilot(False)
            a.destroy()