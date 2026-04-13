import carla
import time
import random


def spawn_traffic(world, blueprint_library):
    vehicles = []

    v1_model = blueprint_library.filter('model3')[0]

    x = random.choice([392.0, 395.5])
    spawn_p1 = carla.Transform(carla.Location(x=x, y=10.0, z=1.5), carla.Rotation(yaw=90.0))
    v1 = world.spawn_actor(v1_model, spawn_p1)
    vehicles.append(v1)

    return vehicles, v1, x


def run(world, blueprint_library):
    run_number = 1
    crossing_line = 315

    print("\nRozpoczynam scenariusz: Nadmierna prędkość (1 pojazd). Naciśnij Ctrl+C, aby wrócić do menu.")

    try:
            initial_speed = random.uniform(40.0, 60.0) / 3.6
            max_speed = random.uniform(100.0, 150.0) / 3.6
            boost_trigger_y = random.uniform(80.0, 180.0)

            vehicles, v1, x = spawn_traffic(world, blueprint_library)

            time.sleep(0.5)

            lane = "Lewy" if x < 394.0 else "Prawy"

            print(f"\n--- PRÓBA NR {run_number} ---")
            print(f"Pojazd - Pas: {lane} | Start: {initial_speed * 3.6:.1f} km/h, Max: {max_speed * 3.6:.1f} km/h")
            print(f"Przyspiesza na Y = {boost_trigger_y:.1f}")

            scenario_active = True

            while scenario_active:
                if not v1.is_alive:
                    break

                loc1 = v1.get_location()

                if loc1.y == 0.0:
                    time.sleep(0.05)
                    continue

                if loc1.y > boost_trigger_y:
                    v1.set_target_velocity(carla.Vector3D(x=0.0, y=max_speed, z=0.0))
                else:
                    v1.set_target_velocity(carla.Vector3D(x=0.0, y=initial_speed, z=0.0))

                if loc1.y > crossing_line:
                    scenario_active = False

                time.sleep(0.05)

            for v in vehicles:
                if v.is_alive:
                    v.destroy()

            run_number += 1
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nPrzerwano scenariusz. Czyszczenie...")
    finally:
        for a in world.get_actors().filter('vehicle.*'):
            a.destroy()