import carla
import time
import random
import math


def spawn_traffic(world, blueprint_library):
    vehicles = []

    v1_model = blueprint_library.filter('model3')[0]
    v2_model = blueprint_library.filter('tt')[0]

    spawn_p1 = carla.Transform(carla.Location(x=392.0, y=35.0, z=1.5), carla.Rotation(yaw=90.0))
    v1 = world.spawn_actor(v1_model, spawn_p1)
    vehicles.append(v1)

    spawn_p2 = carla.Transform(carla.Location(x=395.5, y=15.0, z=1.5), carla.Rotation(yaw=90.0))
    v2 = world.spawn_actor(v2_model, spawn_p2)
    vehicles.append(v2)

    return vehicles, v1, v2


def run(world, blueprint_library):
    run_number = 1
    crossing_line = 315
    target_x = 392
    vehicle_length = 5.0

    print("\nRozpoczynam scenariusz: Zmiana pasa (PRECYZYJNA KINEMATYKA). Naciśnij Ctrl+C, aby wrócić.")

    try:
            vehicles, v1, v2 = spawn_traffic(world, blueprint_library)

            time.sleep(0.5)

            normal_speed = random.uniform(40.0, 50.0) / 3.6

            cut_in_y = random.uniform(80.0, 150.0)

            clearance_distance = random.uniform(1.0, 4.0)

            total_cut_in_distance = clearance_distance + vehicle_length

            victim_target_y = cut_in_y - total_cut_in_distance

            time_to_target = (victim_target_y - 30.0) / normal_speed

            speeder_base_speed = (cut_in_y - 10.0) / time_to_target

            current_speeder_speed = speeder_base_speed

            lateral_speed = random.uniform(4.0, 6.0)

            print(f"\n--- PRÓBA NR {run_number} ---")
            print(f"Manewr na metrze: X = {cut_in_y:.1f} m")
            print(f"Odstęp zderzaków: {clearance_distance:.1f} m")
            print(f"Prędkość dojazdowa pirata: {speeder_base_speed * 3.6:.1f} km/h")

            scenario_active = True
            lane_change_state = 0

            while scenario_active:
                if not v1.is_alive or not v2.is_alive:
                    break

                loc1 = v1.get_location()
                loc2 = v2.get_location()

                if loc1.y == 0.0 or loc2.y == 0.0:
                    time.sleep(0.05)
                    continue

                v1.set_target_velocity(carla.Vector3D(x=0.0, y=normal_speed, z=0.0))

                if lane_change_state == 0:
                    v2.set_target_velocity(carla.Vector3D(x=0.0, y=current_speeder_speed, z=0.0))

                    if loc2.y >= cut_in_y:
                        lane_change_state = 1
                        speed_boost = random.uniform(20.0, 35.0) / 3.6
                        current_speeder_speed += speed_boost
                        print(f" -> Cięcie! Przyspieszam do {current_speeder_speed * 3.6:.1f} km/h")

                        yaw_rad = math.atan2(lateral_speed, current_speeder_speed)
                        yaw_deg = math.degrees(yaw_rad)
                        t2 = v2.get_transform()
                        t2.rotation.yaw = yaw_deg + 90
                        v2.set_transform(t2)

                elif lane_change_state == 1:
                    v2.set_target_velocity(carla.Vector3D(x=-lateral_speed, y=current_speeder_speed, z=0.0))

                    if loc2.x <= target_x:
                        lane_change_state = 2
                        t2 = v2.get_transform()
                        t2.rotation.yaw = 90.0
                        t2.location.x = target_x
                        v2.set_transform(t2)

                elif lane_change_state == 2:
                    v2.set_target_velocity(carla.Vector3D(x=0.0, y=current_speeder_speed, z=0.0))

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
        actors = world.get_actors().filter('vehicle.*')
        for a in actors:
            a.destroy()