import carla
import time
import random


def get_random_car_blueprint(blueprint_library):
    blueprints = [bp for bp in blueprint_library.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) == 4]
    return random.choice(blueprints)


def run(world, blueprint_library):
    crossing_line = 315
    spawn_y = 10.0

    print("\nRozpoczynam scenariusz: Normalny ruch uliczny (Nieskończony). Naciśnij Ctrl+C, aby wrócić.")

    active_vehicles = []
    last_spawn_time = time.time()
    next_spawn_delay = random.uniform(2.0, 4.0)

    try:
        while True:
            current_time = time.time()

            if current_time - last_spawn_time > next_spawn_delay:
                spawn_x = random.choice([392.0, 395.5])
                speed = random.uniform(30.0, 50.0) / 3.6
                bp = get_random_car_blueprint(blueprint_library)
                transform = carla.Transform(
                    carla.Location(x=spawn_x, y=spawn_y, z=1.5),
                    carla.Rotation(yaw=90.0)
                )
                vehicle = world.try_spawn_actor(bp, transform)

                if vehicle is not None:
                    active_vehicles.append((vehicle, speed))
                    next_spawn_delay = random.uniform(1.5, 3.5)
                    last_spawn_time = current_time

            to_remove = []
            for entry in active_vehicles:
                v, speed = entry
                if not v.is_alive:
                    to_remove.append(entry)
                    continue

                loc = v.get_location()

                if loc.y > crossing_line:
                    to_remove.append(entry)
                    continue

                v.set_target_velocity(carla.Vector3D(x=0.0, y=speed, z=0.0))

            for entry in to_remove:
                v, _ = entry
                if v.is_alive:
                    v.destroy()
                if entry in active_vehicles:
                    active_vehicles.remove(entry)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nPrzerwano scenariusz. Czyszczenie ruchu...")
    finally:
        for entry in active_vehicles:
            v, _ = entry
            if v.is_alive:
                v.destroy()
        for a in world.get_actors().filter('vehicle.*'):
            a.destroy()