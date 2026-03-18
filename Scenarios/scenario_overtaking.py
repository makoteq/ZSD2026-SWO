import carla
import time
import random
import math


def spawn_traffic(world, blueprint_library):
    vehicles = []

    v1_model = blueprint_library.filter('model3')[0]
    v2_model = blueprint_library.filter('tt')[0]

    spawn_p1 = carla.Transform(carla.Location(x=40.0, y=5.25, z=2.0), carla.Rotation(yaw=0.0))
    v1 = world.spawn_actor(v1_model, spawn_p1)
    vehicles.append(v1)

    spawn_p2 = carla.Transform(carla.Location(x=10.0, y=5.25, z=2.0), carla.Rotation(yaw=0.0))
    v2 = world.spawn_actor(v2_model, spawn_p2)
    vehicles.append(v2)

    return vehicles, v1, v2


def run(world, blueprint_library):
    run_number = 1
    crossing_line = 180.0

    target_y_right = 5.25
    target_y_left = 1.75
    vehicle_length = 5.0
    lane_width = target_y_right - target_y_left  # 3.5 metra

    print("\nRozpoczynam scenariusz: Wyprzedzanie. Naciśnij Ctrl+C, aby wrócić.")

    try:
        while True:
            vehicles, v1, v2 = spawn_traffic(world, blueprint_library)
            time.sleep(0.5)

            if not v1.is_alive or not v2.is_alive:
                continue

            normal_speed = random.uniform(40.0, 50.0) / 3.6
            speeder_speed = random.uniform(90.0, 130.0) / 3.6
            lateral_speed = random.uniform(4.0, 6.0)

            relative_speed = speeder_speed - normal_speed
            time_to_change_lane = lane_width / lateral_speed

            closing_distance = relative_speed * time_to_change_lane

            pull_out_clearance = random.uniform(2.0, 4.0)

            pull_out_trigger = vehicle_length + closing_distance + pull_out_clearance

            pull_in_clearance = random.uniform(3.0, 6.0)
            pull_in_trigger = vehicle_length + pull_in_clearance

            print(f"\n--- PRÓBA NR {run_number} ---")
            print(f"Pirat: {speeder_speed * 3.6:.1f} km/h | Ofiara: {normal_speed * 3.6:.1f} km/h")
            print(
                f"Pirat zacznie zjeżdżać {pull_out_trigger:.1f} m za ofiarą (aby ominąć ją o {pull_out_clearance:.1f} m).")

            scenario_active = True
            state = 0

            while scenario_active:
                if not v1.is_alive or not v2.is_alive:
                    break

                loc1 = v1.get_location()
                loc2 = v2.get_location()

                if loc1.x == 0.0 or loc2.x == 0.0:
                    time.sleep(0.05)
                    continue

                v1.set_target_velocity(carla.Vector3D(x=normal_speed, y=0.0, z=0.0))
                t1 = v1.get_transform()
                t1.rotation.yaw = 0.0
                v1.set_transform(t1)

                if state == 0:
                    v2.set_target_velocity(carla.Vector3D(x=speeder_speed, y=0.0, z=0.0))
                    t2 = v2.get_transform()
                    t2.rotation.yaw = 0.0
                    v2.set_transform(t2)

                    if loc1.x - loc2.x < pull_out_trigger:
                        state = 1
                        print(" -> Odbicie na lewy pas!")

                elif state == 1:
                    yaw_rad = math.atan2(-lateral_speed, speeder_speed)
                    yaw_deg = math.degrees(yaw_rad)

                    v2.set_target_velocity(carla.Vector3D(x=speeder_speed, y=-lateral_speed, z=0.0))
                    t2 = v2.get_transform()
                    t2.rotation.yaw = yaw_deg
                    v2.set_transform(t2)

                    if loc2.y <= target_y_left:
                        state = 2

                elif state == 2:
                    v2.set_target_velocity(carla.Vector3D(x=speeder_speed, y=0.0, z=0.0))
                    t2 = v2.get_transform()
                    t2.rotation.yaw = 0.0
                    t2.location.y = target_y_left
                    v2.set_transform(t2)

                    if loc2.x - loc1.x > pull_in_trigger:
                        state = 3
                        print(" -> Powrót na prawy pas!")

                elif state == 3:
                    yaw_rad = math.atan2(lateral_speed, speeder_speed)
                    yaw_deg = math.degrees(yaw_rad)

                    v2.set_target_velocity(carla.Vector3D(x=speeder_speed, y=lateral_speed, z=0.0))
                    t2 = v2.get_transform()
                    t2.rotation.yaw = yaw_deg
                    v2.set_transform(t2)

                    if loc2.y >= target_y_right:
                        state = 4

                elif state == 4:
                    v2.set_target_velocity(carla.Vector3D(x=speeder_speed, y=0.0, z=0.0))
                    t2 = v2.get_transform()
                    t2.rotation.yaw = 0.0
                    t2.location.y = target_y_right
                    v2.set_transform(t2)

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
