import carla
#import time
import random
import os
import math

#--------------------------------------------------------configuration--------------------------------------------------
# Lane and spawn configuration
LANE_A_Y = 109.3
LANE_B_Y = 105.7
START_X = 5.0
SPAWN_Z = 0.5
YAW = 0.0
CROSSING_LINE = 260.0
LATERAL_OFFSET_MAX = 0.04

# Speed configuration
MIN_SPEED_KMH = 50.0
MAX_SPEED_KMH = 50.0

# Gap configuration
SPAWN_DELAY_MIN = 2.0
SPAWN_DELAY_MAX = 3.5
MIN_GAP_SAME_LANE_M = 15.0
SAFE_DISTANCE_M = 12.0

#-----------------------------------------------------------------------------------------------------------------------
# Filter to only select passenger cars
def get_random_car_blueprint(blueprint_library):
    """Return a random passenger car blueprint, excluding bikes, vans, and trucks."""
    cars = []
    for bp in blueprint_library.filter("vehicle.*"):
        # Exclude bikes, wheelchairs etc.
        if bp.has_attribute("base_type"):
            if bp.get_attribute("base_type").as_str() != "car":
                continue
        bid = bp.id.lower()
        # Exclude big vans
        excluding = ["firetruck", "ambulance", "bus", "truck", "van", "carlacola", "sprinter"]
        if any(x in bid for x in excluding):
            continue
        cars.append(bp)
    if not cars:
        raise RuntimeError("no cars?")
    return random.choice(cars)


def get_speed_ms(actor):
    """Return the current speed of an actor in m/s."""
    v = actor.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def run(world, blueprint_library, duration_sec=120.0, output_dir=None):
    """Spawn and regulate normal traffic for duration_sec seconds, then save spawn count."""
    spawned_count = 0
    active_vehicles_data = []

    snap = world.get_snapshot()
    start_sim_time = snap.timestamp.elapsed_seconds
    last_spawn_sim_time = start_sim_time
    next_spawn_delay = random.uniform(SPAWN_DELAY_MIN, SPAWN_DELAY_MAX)

    print(f"Scenario: Normal Traffic | Speed: {MIN_SPEED_KMH}-{MAX_SPEED_KMH} km/h")

    try:
        while True:
            world.tick()
            snap = world.get_snapshot()
            sim_time = snap.timestamp.elapsed_seconds
            elapsed = sim_time - start_sim_time

            if elapsed >= duration_sec:
                break

            #Spawn
            if (sim_time - last_spawn_sim_time) >= next_spawn_delay:
                target_lane_y = random.choice([LANE_A_Y, LANE_B_Y])

                can_spawn = True
                for data in active_vehicles_data:
                    v = data["actor"]
                    if v.is_alive:
                        loc = v.get_location()
                        if abs(loc.y - target_lane_y) < 1.5 and abs(loc.x - START_X) < MIN_GAP_SAME_LANE_M:
                            can_spawn = False
                            break

                if can_spawn:
                    spawn_y = target_lane_y + random.uniform(-LATERAL_OFFSET_MAX, LATERAL_OFFSET_MAX)
                    t = carla.Transform(
                        carla.Location(x=START_X, y=spawn_y, z=SPAWN_Z),
                        carla.Rotation(yaw=YAW),
                    )
                    bp = get_random_car_blueprint(blueprint_library)
                    vehicle = world.try_spawn_actor(bp, t)
                    if vehicle:
                        target_speed = random.uniform(MIN_SPEED_KMH, MAX_SPEED_KMH) / 3.6
                        active_vehicles_data.append({
                            "actor": vehicle,
                            "target_speed": target_speed,
                            "lane_y": target_lane_y,
                        })
                        spawned_count += 1
                        last_spawn_sim_time = sim_time
                        next_spawn_delay = random.uniform(SPAWN_DELAY_MIN, SPAWN_DELAY_MAX)

            #Speed regulator
            for data in active_vehicles_data:
                v = data["actor"]
                if not v.is_alive:
                    continue

                my_loc = v.get_location()
                target_speed = data["target_speed"]
                current_applied_speed = target_speed

                closest_dist = 999.0
                front_vehicle_speed = 0.0

                for other_data in active_vehicles_data:
                    other_v = other_data["actor"]
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

            #Clean actors that reach the end of the road
            survivors = []
            for data in active_vehicles_data:
                v = data["actor"]
                if v.is_alive:
                    if v.get_location().x > CROSSING_LINE:
                        v.destroy()
                    else:
                        survivors.append(data)
            active_vehicles_data = survivors

    finally:
        for data in active_vehicles_data:
            try:
                if data["actor"].is_alive:
                    data["actor"].destroy()
            except Exception:
                pass

        log_dir = output_dir if output_dir else "."
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "spawn_count.txt") #dla Vlada do metryk
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(str(spawned_count))