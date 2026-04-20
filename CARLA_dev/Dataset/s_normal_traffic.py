import carla
import time
import random
import math

# Lane and spawn configuration
LANE_A_Y = 109.3
LANE_B_Y = 105.7
START_X = -3.0
SPAWN_Z = 0.5 
YAW = 0.0
CROSSING_LINE = 260.0

# speed configuration
MIN_SPEED_KMH = 30.0
MAX_SPEED_KMH = 50.0

# Gap configuration
SPAWN_DELAY_MIN = 2.0
SPAWN_DELAY_MAX = 3.5
MIN_GAP_SAME_LANE_M = 15.0
SAFE_DISTANCE_M = 12.0  # Distance at which the car starts braking

# Function to get a random car blueprint
def get_random_car_blueprint(blueprint_library):
    cars = [bp for bp in blueprint_library.filter('vehicle.*') 
            if int(bp.get_attribute('number_of_wheels')) == 4]
    cars = [bp for bp in cars if not any(x in bp.id.lower() for x in 
            ['truck', 'bus', 'van', 'ambulance', 'firetruck', 'vespa', 'isetta'])]
    return random.choice(cars)

# Function to get the current speed of an actor in m/s
def get_speed_ms(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def run(world, blueprint_library, duration_sec=120.0, output_dir=None):
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
                
                # Check for space before spawning
                can_spawn = True
                for data in active_vehicles_data:
                    v = data['actor']
                    if v.is_alive:
                        loc = v.get_location()
                        if abs(loc.y - target_lane_y) < 1.5 and abs(loc.x - START_X) < MIN_GAP_SAME_LANE_M:
                            can_spawn = False
                            break
                
                # If there is space, spawn the car directly on the lane center
                if can_spawn:
                    t = carla.Transform(carla.Location(x=START_X, y=target_lane_y, z=SPAWN_Z), carla.Rotation(yaw=YAW))
                    bp = get_random_car_blueprint(blueprint_library)
                    vehicle = world.try_spawn_actor(bp, t)

                    if vehicle:
                        target_speed = random.uniform(MIN_SPEED_KMH, MAX_SPEED_KMH) / 3.6
                        active_vehicles_data.append({
                            'actor': vehicle,
                            'target_speed': target_speed,
                            'lane_y': target_lane_y
                        })
                        last_spawn_time = current_time
            
            for data in active_vehicles_data:
                v = data['actor']
                if not v.is_alive:
                    continue
                
                my_transform = v.get_transform()
                my_loc = my_transform.location
                current_v = v.get_velocity()
                
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
                        if dist < closest_dist:
                            closest_dist = dist
                            front_vehicle_speed = get_speed_ms(other_v)

                # Adjust our speed to match the front vehicle
                if closest_dist < SAFE_DISTANCE_M:
                    current_applied_speed = min(target_speed, front_vehicle_speed * 0.9)
                    if closest_dist < 5.0: 
                        current_applied_speed = 0.0

                
                lookahead_dist = max(8.0, current_applied_speed * 1.5)
                
                dx = lookahead_dist
                dy = data['lane_y'] - my_loc.y
                
                target_yaw_rad = math.atan2(dy, dx)
                target_yaw_deg = math.degrees(target_yaw_rad)

                yaw_error = target_yaw_deg - my_transform.rotation.yaw
                while yaw_error > 180.0: yaw_error -= 360.0
                while yaw_error < -180.0: yaw_error += 360.0

                steer_corr = max(-0.5, min(0.5, yaw_error * 0.05))
                v.apply_control(carla.VehicleControl(steer=steer_corr, throttle=0.0, brake=0.0))

                vx = current_applied_speed * math.cos(target_yaw_rad)
                vy = current_applied_speed * math.sin(target_yaw_rad)


                v.enable_constant_velocity(carla.Vector3D(x=vx, y=vy, z=current_v.z))

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