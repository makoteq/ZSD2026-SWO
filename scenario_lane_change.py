import carla
import time
import random
import math


def spawn_traffic(world, blueprint_library):
    vehicles = []

    v1_model = blueprint_library.filter('model3')[0]
    v2_model = blueprint_library.filter('tt')[0]

    spawn_p1 = carla.Transform(carla.Location(x=30.0, y=5.25, z=2.0), carla.Rotation(yaw=0.0))
    v1 = world.spawn_actor(v1_model, spawn_p1)
    vehicles.append(v1)

    spawn_p2 = carla.Transform(carla.Location(x=10.0, y=1.75, z=2.0), carla.Rotation(yaw=0.0))
    v2 = world.spawn_actor(v2_model, spawn_p2)
    vehicles.append(v2)

    return vehicles, v1, v2


def run(world, blueprint_library):
    run_number = 1
    crossing_line = 180.0
    target_y = 5.25
    vehicle_length = 5.0

    print("\nRozpoczynam scenariusz: Zmiana pasa (PRECYZYJNA KINEMATYKA). Naciśnij Ctrl+C, aby wrócić.")

    try:
        while True:
            vehicles, v1, v2 = spawn_traffic(world, blueprint_library)

            time.sleep(0.5)

            if not v1.is_alive or not v2.is_alive:
                continue

            normal_speed = random.uniform(40.0, 50.0) / 3.6


            cut_in_x = random.uniform(80.0, 150.0)


            clearance_distance = random.uniform(1.0, 4.0)


            total_cut_in_distance = clearance_distance + vehicle_length


            victim_target_x = cut_in_x - total_cut_in_distance


            time_to_target = (victim_target_x - 30.0) / normal_speed


            speeder_base_speed = (cut_in_x - 10.0) / time_to_target


            current_speeder_speed = speeder_base_speed

            lateral_speed = random.uniform(4.0, 6.0)

            print(f"\n--- PRÓBA NR {run_number} ---")
            print(f"Manewr na metrze: X = {cut_in_x:.1f} m")
            print(f"Odstęp zderzaków: {clearance_distance:.1f} m")
            print(f"Prędkość dojazdowa pirata: {speeder_base_speed * 3.6:.1f} km/h")

            scenario_active = True
            lane_change_state = 0

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

                if lane_change_state == 0:
                    v2.set_target_velocity(carla.Vector3D(x=current_speeder_speed, y=0.0, z=0.0))
                    t2 = v2.get_transform()
                    t2.rotation.yaw = 0.0
                    v2.set_transform(t2)

                    if loc2.x >= cut_in_x:
                        lane_change_state = 1
                        speed_boost = random.uniform(20.0, 35.0) / 3.6
                        current_speeder_speed += speed_boost
                        print(f" -> Cięcie! Przyspieszam do {current_speeder_speed * 3.6:.1f} km/h")

                elif lane_change_state == 1:
                    yaw_rad = math.atan2(lateral_speed, current_speeder_speed)
                    yaw_deg = math.degrees(yaw_rad)

                    v2.set_target_velocity(carla.Vector3D(x=current_speeder_speed, y=lateral_speed, z=0.0))
                    t2 = v2.get_transform()
                    t2.rotation.yaw = yaw_deg
                    v2.set_transform(t2)

                    if loc2.y >= target_y:
                        lane_change_state = 2

                elif lane_change_state == 2:
                    v2.set_target_velocity(carla.Vector3D(x=current_speeder_speed, y=0.0, z=0.0))
                    t2 = v2.get_transform()
                    t2.rotation.yaw = 0.0
                    t2.location.y = target_y
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