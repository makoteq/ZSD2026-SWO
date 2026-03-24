import carla
import time
import random


def spawn_traffic(world, blueprint_library):
    vehicles = []

    v1_model = blueprint_library.filter('model3')[0]
    v2_model = blueprint_library.filter('tt')[0]

    lanes = [392.0, 395.5]
    random.shuffle(lanes)
    x_normal, x_speeder = lanes

    spawn_p1 = carla.Transform(carla.Location(x=x_normal, y=30.0, z=1.5), carla.Rotation(yaw=90.0))
    v1 = world.spawn_actor(v1_model, spawn_p1)
    vehicles.append(v1)

    spawn_p2 = carla.Transform(carla.Location(x=x_speeder, y=10.0, z=1.5), carla.Rotation(yaw=90.0))
    v2 = world.spawn_actor(v2_model, spawn_p2)
    vehicles.append(v2)

    return vehicles, v1, v2, x_normal, x_speeder


def run(world, blueprint_library):
    run_number = 1
    crossing_line = 315

    print("\nRozpoczynam scenariusz: Nadmierna prędkość (2 pojazdy). Naciśnij Ctrl+C, aby wrócić do menu.")

    try:
            normal_speed = random.uniform(40.0, 50.0) / 3.6
            speeder_initial_speed = normal_speed
            speeder_max_speed = random.uniform(90.0, 140.0) / 3.6
            boost_trigger_y = random.uniform(80.0, 180.0)

            vehicles, v1, v2, x_normal, x_speeder = spawn_traffic(world, blueprint_library)

            time.sleep(0.5)

            lane_n = "Lewy" if x_normal < 394.0 else "Prawy"
            lane_s = "Lewy" if x_speeder < 394.0 else "Prawy"

            print(f"\n--- PRÓBA NR {run_number} ---")
            print(f"V1 (Normalny) - Pas: {lane_n} | {normal_speed * 3.6:.1f} km/h")
            print(f"V2 (Pirat)    - Pas: {lane_s} | Start: {speeder_initial_speed * 3.6:.1f} km/h, Max: {speeder_max_speed * 3.6:.1f} km/h")
            print(f"V2 zaczyna przyspieszać na Y = {boost_trigger_y:.1f}")

            scenario_active = True

            while scenario_active:
                if not v1.is_alive or not v2.is_alive:
                    break

                loc1 = v1.get_location()
                loc2 = v2.get_location()

                if loc1.y == 0.0 or loc2.y == 0.0:
                    time.sleep(0.05)
                    continue

                v1.set_target_velocity(carla.Vector3D(x=0.0, y=normal_speed, z=0.0))

                if loc2.y > boost_trigger_y:
                    v2.set_target_velocity(carla.Vector3D(x=0.0, y=speeder_max_speed, z=0.0))
                else:
                    v2.set_target_velocity(carla.Vector3D(x=0.0, y=speeder_initial_speed, z=0.0))

                if loc1.y > crossing_line and loc2.y > crossing_line:
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