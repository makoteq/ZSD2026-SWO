import carla
import time
import random
import math


def spawn_traffic(world, blueprint_library):
    vehicles = []

    v1_model = blueprint_library.filter('model3')[0]
    v2_model = blueprint_library.filter('tt')[0]

    spawn_p1 = carla.Transform(carla.Location(x=392.0, y=40.0, z=1.5), carla.Rotation(yaw=90.0))
    v1 = world.spawn_actor(v1_model, spawn_p1)
    vehicles.append(v1)

    spawn_p2 = carla.Transform(carla.Location(x=392.0, y=15.0, z=1.5), carla.Rotation(yaw=90.0))
    v2 = world.spawn_actor(v2_model, spawn_p2)
    vehicles.append(v2)

    return vehicles, v1, v2


def run(world, blueprint_library):
    run_number = 1
    crossing_line = 315

    x_right = 392.0
    x_left = 395.5
    vehicle_length = 5.0
    lane_width = abs(x_left - x_right)

    print("\nRozpoczynam scenariusz: Wyprzedzanie. Naciśnij Ctrl+C, aby wrócić.")

    try:
            vehicles, v1, v2 = spawn_traffic(world, blueprint_library)
            time.sleep(0.5)

            normal_speed = random.uniform(40.0, 50.0) / 3.6
            speeder_speed = random.uniform(90.0, 130.0) / 3.6
            lateral_speed = random.uniform(4.0, 6.0)

            time_to_change_lane = lane_width / lateral_speed
            closing_distance = (speeder_speed - normal_speed) * time_to_change_lane
            pull_out_clearance = random.uniform(2.0, 4.0)
            pull_out_trigger = vehicle_length + closing_distance + pull_out_clearance
            pull_in_clearance = random.uniform(3.0, 6.0)
            pull_in_trigger = vehicle_length + pull_in_clearance

            print(f"\n--- PRÓBA NR {run_number} ---")
            print(f"Pirat: {speeder_speed * 3.6:.1f} km/h | Ofiara: {normal_speed * 3.6:.1f} km/h")
            print(f"Wyprzedzanie gdy dystans < {pull_out_trigger:.1f} m")

            state = 0
            active = True

            while active:
                if not v1.is_alive or not v2.is_alive:
                    break

                loc1 = v1.get_location()
                loc2 = v2.get_location()

                if loc1.y == 0.0 or loc2.y == 0.0:
                    time.sleep(0.05)
                    continue

                v1.set_target_velocity(carla.Vector3D(x=0.0, y=normal_speed, z=0.0))

                if state == 0:
                    v2.set_target_velocity(carla.Vector3D(x=0.0, y=speeder_speed, z=0.0))

                    if loc1.y - loc2.y < pull_out_trigger:
                        state = 1
                        print(" -> Odbicie na lewy pas!")
                        yaw_rad = math.atan2(lateral_speed, speeder_speed)
                        yaw_deg = math.degrees(yaw_rad)
                        t2 = v2.get_transform()
                        t2.rotation.yaw = 90.0 - yaw_deg
                        v2.set_transform(t2)

                elif state == 1:
                    v2.set_target_velocity(carla.Vector3D(x=lateral_speed, y=speeder_speed, z=0.0))

                    if loc2.x >= x_left:
                        state = 2
                        t2 = v2.get_transform()
                        t2.rotation.yaw = 90.0
                        t2.location.x = x_left
                        v2.set_transform(t2)

                elif state == 2:
                    v2.set_target_velocity(carla.Vector3D(x=0.0, y=speeder_speed, z=0.0))

                    if loc2.y - loc1.y > pull_in_trigger:
                        state = 3
                        print(" -> Powrót na prawy pas!")
                        yaw_rad = math.atan2(lateral_speed, speeder_speed)
                        yaw_deg = math.degrees(yaw_rad)
                        t2 = v2.get_transform()
                        t2.rotation.yaw = 90.0 + yaw_deg
                        v2.set_transform(t2)

                elif state == 3:
                    v2.set_target_velocity(carla.Vector3D(x=-lateral_speed, y=speeder_speed, z=0.0))

                    if loc2.x <= x_right:
                        state = 4
                        t2 = v2.get_transform()
                        t2.rotation.yaw = 90.0
                        t2.location.x = x_right
                        v2.set_transform(t2)

                elif state == 4:
                    v2.set_target_velocity(carla.Vector3D(x=0.0, y=speeder_speed, z=0.0))

                if loc1.y > crossing_line:
                    active = False

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