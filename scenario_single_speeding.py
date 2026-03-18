import carla
import time
import random


def spawn_traffic(world, blueprint_library):
    vehicles = []

    v1_model = blueprint_library.filter('model3')[0]


    y_speeder = random.choice([1.75, 5.25])


    spawn_p1 = carla.Transform(carla.Location(x=10.0, y=y_speeder, z=0.05), carla.Rotation(yaw=0.0))
    v1 = world.spawn_actor(v1_model, spawn_p1)
    vehicles.append(v1)

    return vehicles, v1, y_speeder


def run(world, blueprint_library):
    run_number = 1
    crossing_line = 180.0

    print("\nRozpoczynam scenariusz: Nadmierna prędkość (1 pojazd). Naciśnij Ctrl+C, aby wrócić do menu.")

    try:
        while True:
            initial_speed = random.uniform(40.0, 60.0) / 3.6
            max_speed = random.uniform(100.0, 150.0) / 3.6

            boost_trigger_x = random.uniform(50.0, 120.0)

            vehicles, v1, y_speeder = spawn_traffic(world, blueprint_library)

            time.sleep(0.5)

            lane_str = "Lewy" if y_speeder < 3.0 else "Prawy"

            print(f"\n--- PRÓBA NR {run_number} ---")
            print(f"Pojazd - Pas: {lane_str} | Start: {initial_speed * 3.6:.1f} km/h, Max: {max_speed * 3.6:.1f} km/h")
            print(f"Przyspiesza na X = {boost_trigger_x:.1f}")

            scenario_active = True

            while scenario_active:
                if not v1.is_alive:
                    break

                loc1 = v1.get_location()

                if loc1.x == 0.0:
                    time.sleep(0.05)
                    continue

                if loc1.x > boost_trigger_x:
                    v1.set_target_velocity(carla.Vector3D(x=max_speed, y=0.0, z=0.0))
                else:
                    v1.set_target_velocity(carla.Vector3D(x=initial_speed, y=0.0, z=0.0))

                if loc1.x > crossing_line:
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
        actors = world.get_actors().filter('vehicle.*')
        for a in actors:
            a.destroy()