import os
import csv
import math
import time
import carla
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LANE_A_Y = 109.5
LANE_B_Y = 105.5
START_X = -3.0
SPAWN_Z = 0.8
YAW = 0.0
CROSSING_LINE = 260.0

# Speed ranges for randomization (max 50 km/h)
SLOW_MIN_KPH = 20.0
SLOW_MAX_KPH = 28.0

INITIAL_MIN_KPH = 20.0
INITIAL_MAX_KPH = 28.0

TARGET_MIN_KPH = 42.0
TARGET_MAX_KPH = 50.0

BOOST_TRIGGER_X = 80.0
# logs for speed testing ------------------------------------------------
def save_speed_profile(speed_samples, out_dir, prefix="ego"):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{prefix}_speed_profile.csv")
    png_path = os.path.join(out_dir, f"{prefix}_speed_profile.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "speed_kmh"])
        w.writerows(speed_samples)

    t = [x[0] for x in speed_samples]
    s = [x[1] for x in speed_samples]

    plt.figure(figsize=(10, 4))
    plt.plot(t, s, linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [km/h]")
    plt.title(f"Vehicle speed profile - {prefix}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"[SPEED] CSV: {csv_path}")
    print(f"[SPEED] PNG: {png_path}")

# utility ---------------------------------------------------
def destroy_all_vehicles(world):
    # remove all vehicles from the world
    for a in world.get_actors().filter('vehicle.*'):
        if a.is_alive:
            a.destroy()

def speed_ms(vehicle):
    # calculate speed in meters per second
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

def speed_kph(vehicle):
    # calculate speed in kilometers per hour
    v = vehicle.get_velocity()
    ms = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
    return ms*3.6

def apply_speed(vehicle, target_speed, steer=0.0):
    target_speed = target_speed/3.6
    v = speed_ms(vehicle)
    delta = target_speed - v

    reg_throttle = 0.8
    max_throttle = 0.85
    reg_brake = 0.25
    max_brake = 0.4

    # apply throttle or brake based on speed difference
    if delta >= 0.5:
        throttle = min(max_throttle, reg_throttle*delta)
        brake = 0.0
    elif delta <= -0.5:
        throttle = 0.0
        brake = min(max_brake, reg_brake*abs(delta))
    else:
        brake = 0.0
        throttle = 0.0

    vehicle.apply_control(carla.VehicleControl(
        throttle=throttle,
        brake=brake,
        steer=steer,
        hand_brake=False,
        manual_gear_shift=False
    ))

def steer_to_y(y_target, y_current, current_yaw, kp=0.08, k_yaw=0.03, limit=0.18):
    y_err = y_target - y_current
    
    yaw = current_yaw
    # fix angle to be between -180 and 180 degrees
    while yaw > 180.0: yaw -= 360.0
    while yaw < -180.0: yaw += 360.0
    
    # calculate steering to stay on lane and prevent oscillations
    steer = kp * y_err - (k_yaw * yaw)
    steer = max(-limit, min(limit, steer))
    
    return steer


def run(world, blueprint_library, duration_sec=30.0, output_dir=None):
    destroy_all_vehicles(world)

    v1_model = blueprint_library.find('vehicle.dodge.charger_2020')
    
    y_slow = LANE_A_Y
    y_fast = LANE_B_Y

    t_slow = carla.Transform(carla.Location(x=START_X, y=y_slow, z=SPAWN_Z), carla.Rotation(yaw=YAW))
    t_fast = carla.Transform(carla.Location(x=START_X, y=y_fast, z=SPAWN_Z), carla.Rotation(yaw=YAW))

    # spawn slow car
    v_slow = world.try_spawn_actor(v1_model, t_slow)
    if v_slow is None:
        print("[SPAWN ERROR] v_slow")
        return
        
    # randomize speeds for this run
    slow_kph = random.uniform(SLOW_MIN_KPH, SLOW_MAX_KPH)
    initial_fast_kph = random.uniform(INITIAL_MIN_KPH, INITIAL_MAX_KPH)
    target_fast_kph = random.uniform(TARGET_MIN_KPH, TARGET_MAX_KPH)
    spawn_delay = random.uniform(1.5, 2.5)

    # set initial speed for slow car
    v_slow.set_target_velocity(carla.Vector3D(x=slow_kph / 3.6, y=0.0, z=0.0))

    v_fast = None
    fast_spawned = False
    
    print(f"[speeding_with_ref] slow={slow_kph:.1f}, fast_start={initial_fast_kph:.1f}, target={target_fast_kph:.1f}, delay={spawn_delay:.1f}s")
    
    t0 = time.time()
    speed_fast = []
    speed_slow = []

    try:
        scenario_active = True
        while scenario_active:
            # check if cars exist
            if not v_slow.is_alive:
                break
                
            if fast_spawned and not v_fast.is_alive:
                break

            now = time.time()
            t_rel = now - t0

            # end scenario if time is up
            if duration_sec is not None and t_rel >= duration_sec:
                break

            loc_slow = v_slow.get_location()
            # end scenario if slow car crosses the line
            if loc_slow.x > CROSSING_LINE:
                scenario_active = False
                break

            speed_slow.append((t_rel, speed_kph(v_slow)))
            
            # keep slow car on its lane
            steer_slow = steer_to_y(y_slow, loc_slow.y, v_slow.get_transform().rotation.yaw)
            apply_speed(v_slow, slow_kph, steer=steer_slow)

            # logic for fast car
            if not fast_spawned:
                # spawn fast car after delay
                if t_rel >= spawn_delay:
                    v_fast = world.try_spawn_actor(v1_model, t_fast)
                    if v_fast is None:
                        print("[SPAWN ERROR] v_fast")
                        break
                    
                    # set initial speed for fast car
                    v_fast.set_target_velocity(carla.Vector3D(x=initial_fast_kph / 3.6, y=0.0, z=0.0))
                    fast_spawned = True
            else:
                loc_fast = v_fast.get_location()
                speed_fast.append((t_rel, speed_kph(v_fast)))

                # end scenario if fast car crosses the line
                if loc_fast.x > CROSSING_LINE:
                    scenario_active = False
                    break

                # keep fast car on its lane and accelerate
                steer_fast = steer_to_y(y_fast, loc_fast.y, v_fast.get_transform().rotation.yaw)
                if loc_fast.x < BOOST_TRIGGER_X:
                    apply_speed(v_fast, initial_fast_kph, steer=steer_fast)
                else:
                    apply_speed(v_fast, target_fast_kph, steer=steer_fast)

            time.sleep(0.05)
            
    finally:
        destroy_all_vehicles(world)
        # save charts
        if output_dir is not None:
            if len(speed_fast) > 1:
                save_speed_profile(speed_fast, output_dir, prefix="speeding_fast")
            if len(speed_slow) > 1:
                save_speed_profile(speed_slow, output_dir, prefix="speeding_slow")