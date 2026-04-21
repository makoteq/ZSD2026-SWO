import carla
import time
import random
import os

# Lane and spawn configuration
LANE_A_Y = 109.3
LANE_B_Y = 105.7
START_X = -3.0
SPAWN_Z = 0.5
YAW = 0.0
CROSSING_LINE = 260.0
LATERAL_OFFSET_MAX = 0.1

# speed configuration
MIN_SPEED_KMH = 30.0
MAX_SPEED_KMH = 50.0

# Gap configuration
SPAWN_DELAY_MIN = 2.0
SPAWN_DELAY_MAX = 3.5
MIN_GAP_SAME_LANE_M = 15.0
SAFE_DISTANCE_M = 12.0  # Distance at which the car starts braking


# Function to get a random car blueprint - but better
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


# Function to get the current speed of an actor in m/s
def get_speed_ms(actor):
    v = actor.get_velocity()
    import math
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def run(world, blueprint_library, duration_sec=120.0, output_dir=None):
    spawned_count = 0
    active_vehicles_data = []
    start_time = time.time()
    last_spawn_time = 0

    print(f"Scenariusz: Normal Traffic (Manual Safety) | Prędkość: {MIN_SPEED_KMH}-{MAX_SPEED_KMH} km/h")

    try:
        while (time.time() - start_time) < duration_sec:
            current_time = time.time()

            # Spawning new cars
            if current_time - last_spawn_time >= random.uniform(SPAWN_DELAY_MIN, SPAWN_DELAY_MAX):
                target_lane_y = random.choice([LANE_A_Y, LANE_B_Y])

                # Chceck for space before spawning
                can_spawn = True
                for data in active_vehicles_data:
                    v = data['actor']
                    if v.is_alive:
                        loc = v.get_location()
                        if abs(loc.y - target_lane_y) < 1.5 and abs(loc.x - START_X) < MIN_GAP_SAME_LANE_M:
                            can_spawn = False
                            break
                # If there is space, spawn the car
                if can_spawn:
                    spawn_y = target_lane_y + random.uniform(-LATERAL_OFFSET_MAX, LATERAL_OFFSET_MAX)
                    t = carla.Transform(carla.Location(x=START_X, y=spawn_y, z=SPAWN_Z), carla.Rotation(yaw=YAW))
                    bp = get_random_car_blueprint(blueprint_library)
                    vehicle = world.try_spawn_actor(bp, t)

                    # If the car was successfully spawned, add it to the active vehicles list with its target speed and lane information
                    if vehicle:
                        target_speed = random.uniform(MIN_SPEED_KMH, MAX_SPEED_KMH) / 3.6
                        active_vehicles_data.append({
                            'actor': vehicle,
                            'target_speed': target_speed,
                            'lane_y': target_lane_y
                        })
                        spawned_count += 1
                        last_spawn_time = current_time

            for data in active_vehicles_data:
                v = data['actor']
                if not v.is_alive:
                    continue

                my_loc = v.get_location()
                target_speed = data['target_speed']
                current_applied_speed = target_speed

                # Check for the closest vehicle in front on the same lane
                closest_dist = 999.0
                front_vehicle_speed = 0.0

                for other_data in active_vehicles_data:
                    other_v = other_data['actor']
                    if other_v.id == v.id or not other_v.is_alive:
                        continue
                    other_loc = other_v.get_location()

                    # Check if the other vehicle is in the same lane and in front of us
                    if abs(other_loc.y - my_loc.y) < 2.0 and other_loc.x > my_loc.x:
                        dist = other_loc.x - my_loc.x
                        # If the vehicle in front is closer than the closest one we've seen so far, update the closest distance and the speed of the front vehicle
                        if dist < closest_dist:
                            closest_dist = dist
                            front_vehicle_speed = get_speed_ms(other_v)

                # If the closest vehicle in front is within the safe distance, adjust our speed to match it or stop if it's very close
                if closest_dist < SAFE_DISTANCE_M:
                    # If the front vehicle is within the safe distance, we want to match its speed but also maintain a safety margin
                    current_applied_speed = min(target_speed, front_vehicle_speed * 0.9)
                    if closest_dist < 5.0:  # Hard stop if the front vehicle is very close
                        current_applied_speed = 0.0

                v.enable_constant_velocity(carla.Vector3D(x=current_applied_speed, y=0.0, z=0.0))

            # Remove vehicles that have crossed the line or are no longer alive
            survivors = []
            for data in active_vehicles_data:
                v = data['actor']
                if v.is_alive:
                    if v.get_location().x > CROSSING_LINE:
                        v.destroy()
                    else:
                        survivors.append(data)
            active_vehicles_data = survivors

            time.sleep(0.05)

    finally:
        for data in active_vehicles_data:
            if data['actor'].is_alive:
                data['actor'].destroy()
            log_dir = output_dir if output_dir else "."
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "spawn_count.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(str(spawned_count))