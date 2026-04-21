import carla
import time
import random
import math
import os
import csv
import matplotlib.pyplot as plt

LANE_A_Y = 109.5
LANE_B_Y = 105.5
START_X = -3.0
SPAWN_Z = 0.8
YAW = 0.0
CROSSING_LINE = 260.0
BOOST_TRIGGER_X = 80.0
DT = 0.05

# strict speed limits (max 50 km/h)
SLOW_MIN_KPH = 20.0
SLOW_MAX_KPH = 25.0
OVERTAKE_MAX_KPH = 50.0

# passing parameters
PASS_GAP_X = 12.0  # gap before cutting in (smaller for aggressive maneuver at lower speeds)
CUTIN_Y_TOL = 0.15

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
    plt.title("Vehicle speed profile")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

def destroy_all_vehicles(world):
    for a in world.get_actors().filter('vehicle.*'):
        if a.is_alive:
            a.destroy()

def speed_ms(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

def speed_kph(vehicle):
    v = vehicle.get_velocity()
    ms = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
    return ms*3.6

def apply_speed(vehicle, target_speed, steer = 0.0):
    target_speed = target_speed/3.6
    v = speed_ms(vehicle)
    delta = target_speed - v

    reg_throttle = 0.8
    max_throttle = 0.85
    reg_brake = 0.25
    max_brake = 0.4

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
        throttle=throttle, brake=brake, steer=steer,
        hand_brake=False, manual_gear_shift=False
    ))

def steer_to_y(y_target, y_current, current_yaw, kp=0.08, k_yaw=0.03, limit=0.18):
    y_err = y_target - y_current
    yaw = current_yaw
    while yaw > 180.0: yaw -= 360.0
    while yaw < -180.0: yaw += 360.0
    
    steer = kp * y_err - (k_yaw * yaw)
    steer = max(-limit, min(limit, steer))
    return steer

# main --------------------------------------------------------------
def run(world, blueprint_library, duration_sec=30.0, output_dir=None):
    destroy_all_vehicles(world)

    bp_fast = blueprint_library.find('vehicle.dodge.charger_2020')
    bp_slow = blueprint_library.find('vehicle.dodge.charger_2020')

    y_slow = LANE_A_Y
    y_fast = LANE_B_Y

    # spawn slow car slightly ahead
    t_slow = carla.Transform(carla.Location(x=START_X+7.5, y=y_slow, z=SPAWN_Z), carla.Rotation(yaw=YAW))
    t_fast = carla.Transform(carla.Location(x=START_X, y=y_fast, z=SPAWN_Z), carla.Rotation(yaw=YAW))

    v_slow = world.try_spawn_actor(bp_slow, t_slow)
    v_fast = world.try_spawn_actor(bp_fast, t_fast)

    if v_slow is None or v_fast is None:
        print("[SPAWN ERROR] overtake")
        destroy_all_vehicles(world)
        return

    # initialize speeds ensuring fast car stays behind initially
    slow_cruise = random.uniform(SLOW_MIN_KPH, SLOW_MAX_KPH)
    fast_cruise = slow_cruise 
    overtake_target = min(50.0, random.uniform(45.0, OVERTAKE_MAX_KPH))

    print(f"[overtake] slow={slow_cruise:.1f} fast_start={fast_cruise:.1f} overtake={overtake_target:.1f}")

    t0 = time.time()
    speed_fast = []
    speed_slow = []
    phase = "cruise"

    try:
        while True:
            if (not v_fast.is_alive) or (not v_slow.is_alive):
                break

            now = time.time()
            t_rel = now - t0

            loc_fast = v_fast.get_location()
            loc_slow = v_slow.get_location()
            yaw_fast = v_fast.get_transform().rotation.yaw

            speed_fast.append((t_rel, speed_kph(v_fast)))
            speed_slow.append((t_rel, speed_kph(v_slow)))

            if duration_sec is not None and t_rel >= duration_sec:
                break
            if loc_fast.x > CROSSING_LINE or loc_slow.x > CROSSING_LINE:
                break

            # Phase 1: Keep distance and wait for trigger
            if phase == "cruise":
                gap = loc_slow.x - loc_fast.x
                # prevent rear-ending
                if gap < 6.0:
                    current_fast_target = slow_cruise - 5.0
                else:
                    current_fast_target = fast_cruise

                apply_speed(v_fast, current_fast_target, steer=0.0)
                apply_speed(v_slow, slow_cruise, steer=0.0)
                
                # initiate overtake when line is crossed
                if loc_fast.x > BOOST_TRIGGER_X:
                    phase = "boost"

            # Phase 2: Accelerate to overtake speed
            elif phase == "boost":
                apply_speed(v_fast, overtake_target, steer=0.0)
                apply_speed(v_slow, slow_cruise, steer=0.0)

                # check if enough space to cut in
                if (loc_fast.x - loc_slow.x) >= PASS_GAP_X:
                    phase = "cut_in"

            # Phase 3: Steer into slow car's lane
            elif phase == "cut_in":
                steer_cmd = steer_to_y(
                    y_target=y_slow, y_current=loc_fast.y, current_yaw=yaw_fast,
                    kp=0.05, k_yaw=0.02, limit=0.15
                )
                apply_speed(v_fast, overtake_target, steer=steer_cmd)
                apply_speed(v_slow, slow_cruise, steer=0.0)

                if abs(y_slow - loc_fast.y) < CUTIN_Y_TOL:
                    # stabilize steering shortly
                    v_fast.apply_control(carla.VehicleControl(
                        throttle=0.12, brake=0.0, steer=0.0,
                        hand_brake=False, manual_gear_shift=False
                    ))
                    phase = "stabilize"

            # Phase 4: Maintain lane and slow down slightly
            elif phase == "stabilize":
                steer_cmd = steer_to_y(
                    y_target=y_slow, y_current=loc_fast.y, current_yaw=yaw_fast,
                    kp=0.05, k_yaw=0.03, limit=0.07
                )
                apply_speed(v_fast, min(50.0, overtake_target - 5.0), steer=steer_cmd)
                apply_speed(v_slow, slow_cruise, steer=0.0)

            # async mode delay
            time.sleep(DT)

    finally:
        if output_dir is not None:
            if len(speed_fast) > 1:
                save_speed_profile(speed_fast, output_dir, prefix="overtake_fast")
            if len(speed_slow) > 1:
                save_speed_profile(speed_slow, output_dir, prefix="overtake_slow")
        destroy_all_vehicles(world)