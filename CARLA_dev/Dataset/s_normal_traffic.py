import carla
import time
import random
import os
import math

LANE_A_Y = 109.3
LANE_B_Y = 105.7
START_X = -3.0
SPAWN_Z = 0.5
YAW = 0.0
CROSSING_LINE = 260.0
LATERAL_OFFSET_MAX = 0.1

MIN_SPEED_KMH = 50.0
MAX_SPEED_KMH = 50.0

SPAWN_DELAY_MIN = 4.5
SPAWN_DELAY_MAX = 7.5
MIN_GAP_SAME_LANE_M = 30.0
SAFE_DISTANCE_M = 12.0


def get_random_car_blueprint(blueprint_library):
    cars = []
    for bp in blueprint_library.filter('vehicle.*'):
        if bp.has_attribute('base_type'):
            if bp.get_attribute('base_type').as_str() != 'car':
                continue

        bid = bp.id.lower()
        excluding = ['firetruck', 'ambulance', 'bus', 'truck', 'van', 'carlacola', 'sprinter']
        if any(x in bid for x in excluding):
            continue
        cars.append(bp)

    if not cars:
        raise RuntimeError("no cars?")
    return random.choice(cars)


def get_speed_ms(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def run(world, blueprint_library, duration_sec=120.0, output_dir=None):
    spawned_count = 0
    active_vehicles_data = []

    world.tick()
    start_sim_time = world.get_snapshot().timestamp.elapsed_seconds
    last_spawn_time = start_sim_time

    print(f"Scenariusz: Normal Traffic (Manual Safety) | Prędkość: {MIN_SPEED_KMH}-{MAX_SPEED_KMH} km/h")

    try:
        while True:
            world.tick()
            sim_time = world.get_snapshot().timestamp.elapsed_seconds
            if (sim_time - start_sim_time) >= duration_sec:
                break

            if sim_time - last_spawn_time >= random.uniform(SPAWN_DELAY_MIN, SPAWN_DELAY_MAX):
                target_lane_y = random.choice([LANE_A_Y, LANE_B_Y])

                can_spawn = True
                for data in active_vehicles_data:
                    v = data['actor']
                    if v.is_alive:
                        loc = v.get_location()
                        if abs(loc.y - target_lane_y) < 1.5 and abs(loc.x - START_X) < MIN_GAP_SAME_LANE_M:
                            can_spawn = False
                            break
                if can_spawn:
                    spawn_y = target_lane_y + random.uniform(-LATERAL_OFFSET_MAX, LATERAL_OFFSET_MAX)
                    t = carla.Transform(carla.Location(x=START_X, y=spawn_y, z=SPAWN_Z), carla.Rotation(yaw=YAW))
                    bp = get_random_car_blueprint(blueprint_library)
                    vehicle = world.try_spawn_actor(bp, t)

                    if vehicle:
                        target_speed = random.uniform(MIN_SPEED_KMH, MAX_SPEED_KMH) / 3.6
                        active_vehicles_data.append({
                            'actor': vehicle,
                            'target_speed': target_speed,
                            'lane_y': target_lane_y
                        })
                        spawned_count += 1
                        last_spawn_time = sim_time

            for data in active_vehicles_data:
                v = data['actor']
                if not v.is_alive:
                    try:
                        v.destroy()
                    except Exception:
                        pass
                    continue

                my_loc = v.get_location()
                target_speed = data['target_speed']
                current_applied_speed = target_speed

                closest_dist = 999.0
                front_vehicle_speed = 0.0

                for other_data in active_vehicles_data:
                    other_v = other_data['actor']
                    if other_v.id == v.id or not other_v.is_alive:
                        continue
                    other_loc = other_v.get_location()

                    if abs(other_loc.y - my_loc.y) < 2.0 and other_loc.x > my_loc.x:
                        dist = other_loc.x - my_loc.x
                        if dist < closest_dist:
                            closest_dist = dist
                            front_vehicle_speed = get_speed_ms(other_v)

                if closest_dist < SAFE_DISTANCE_M:
                    current_applied_speed = min(target_speed, front_vehicle_speed * 0.9)
                    if closest_dist < 5.0:
                        current_applied_speed = 0.0

                v.enable_constant_velocity(carla.Vector3D(x=current_applied_speed, y=0.0, z=0.0))

            survivors = []
            for data in active_vehicles_data:
                v = data['actor']
                if v.is_alive:
                    if v.get_location().x > CROSSING_LINE:
                        v.destroy()
                    else:
                        survivors.append(data)
            active_vehicles_data = survivors

    finally:
        for data in active_vehicles_data:
            if data['actor'].is_alive:
                data['actor'].destroy()

        log_dir = output_dir if output_dir else "."
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "spawn_count.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(str(spawned_count))