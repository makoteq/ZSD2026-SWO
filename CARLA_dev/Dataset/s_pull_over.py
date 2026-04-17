#anomaly
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


CRUISE_MIN_KPH = 40.0
CRUISE_MAX_KPH = 50.0

PULL_START_X_MIN = 60.0
PULL_START_X_MAX = 130.0
PULL_DURATION_SEC = 2.2

MIN_SHIFT = 0.2
MAX_SHIFT = 0.5

DT = 0.05

#logs for speed testing ------------------------------------------------
def save_speed_profile(speed_samples, out_dir, prefix="ego"):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{prefix}_speed_profile.csv")
    png_path = os.path.join(out_dir, f"{prefix}_speed_profile.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:    #save csv
        w = csv.writer(f)
        w.writerow(["t_sec", "speed_kmh"])
        w.writerows(speed_samples)

    #plot
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

    print(f"[SPEED] CSV: {csv_path}")
    print(f"[SPEED] PNG: {png_path}")

#utility ---------------------------------------------------
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

#speed manipulation -----------------------------------------
def apply_speed(vehicle, target_speed, steer = 0.0):
    target_speed = target_speed/3.6                 #kph to mps
    v = speed_ms(vehicle)
    delta = target_speed - v

    reg_throttle = 0.8        #regulator gain
    max_throttle = 0.85         #max throttle
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

    #apply variables to carla car
    vehicle.apply_control(carla.VehicleControl(
        throttle=throttle,
        brake=brake,
        steer=steer,
        hand_brake=False,
        manual_gear_shift=False
    ))

def brake_to_stop(vehicle, steer = 0.0):
    v = speed_kph(vehicle)
    if v > 20:
        brake = 0.25
    elif v > 8:
        brake = 0.35
    elif v > 2:
        brake = 0.45
    else:
        brake = 0.60

    vehicle.apply_control(carla.VehicleControl(
        throttle=0.0,
        brake=brake,
        steer=steer,
        hand_brake=False,
        manual_gear_shift=False
    ))

# main --------------------------------------------------------------

def run(world, blueprint_library, duration_sec=30.0, output_dir=None):
    cruise_kph = random.uniform(CRUISE_MIN_KPH, CRUISE_MAX_KPH)
    pull_start_x = random.uniform(PULL_START_X_MIN, PULL_START_X_MAX)
    shoulder_shift = random.uniform(MIN_SHIFT, MAX_SHIFT)

    v1_model = blueprint_library.find('vehicle.dodge.charger_2020')
    lane_y = random.choice([LANE_A_Y, LANE_B_Y])
    #lane_y = LANE_B_Y
    if lane_y == LANE_B_Y:
        target_y = lane_y - shoulder_shift
    else:
        target_y = lane_y + shoulder_shift

    print(f"[pull_over] cruise={cruise_kph:.1f} km/h, pull_start_x={pull_start_x:.1f}, target_y={target_y:.2f}")

    t1 = carla.Transform(carla.Location(x=START_X, y=lane_y, z=SPAWN_Z), carla.Rotation(yaw=YAW))
    v1 = world.try_spawn_actor(v1_model, t1)
    v1.set_target_velocity(carla.Vector3D(x=cruise_kph / 3.6, y=0.0, z=0.0))

    if v1 is None:
        print("[SPAWN ERROR] pull_over")
        return

    speed_samples = []
    t0 = time.time()

    try:
        #quick phase to meet target speed
        warmup_t0 = time.time()
        while time.time() - warmup_t0 < 2.0:
            apply_speed(v1, cruise_kph, steer=0.0)
            speed_samples.append((time.time() - t0, speed_kph(v1)))
            time.sleep(DT)

        phase = "cruise"
        pull_t0 = None
        scenario_active = True

        while scenario_active:
            if not v1.is_alive:
                break

            loc1 = v1.get_location()
            v_kph = speed_kph(v1)
            speed_samples.append((time.time() - t0, v_kph))

            #timeout because the car stops and never reaches (at least it shouldnt) the finish line
            if duration_sec is not None and (time.time() - t0) >= duration_sec:
                break

            if loc1.x > CROSSING_LINE:
                scenario_active = False
                break

            if phase == "cruise":
                apply_speed(v1, cruise_kph, steer=0.0)
                if loc1.x >= pull_start_x:
                    phase = "pull"
                    pull_t0 = time.time()

            elif phase == "pull":
                #steer to reach target disp
                y_err = target_y - loc1.y
                steer_cmd = max(-0.22, min(0.22, 0.18 * y_err))
                apply_speed(v1, cruise_kph, steer=steer_cmd)

                if (time.time() - pull_t0) >= PULL_DURATION_SEC or abs(y_err) < 0.25:
                    phase = "straighten"

            elif phase == "straighten":
                apply_speed(v1, max(25.0, cruise_kph * 0.8), steer=0.0)
                if abs(v1.get_location().y - target_y) < 0.35:
                    phase = "brake"

            elif phase == "brake":
                brake_to_stop(v1, steer=0.0)
                if v_kph < 0.8:
                    phase = "stop"

            elif phase == "stop":
                v1.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=1.0, steer=0.0, hand_brake=True
                ))
                time.sleep(1.0)
                scenario_active = False
                break

            time.sleep(DT)

    finally:
        destroy_all_vehicles(world)
        if output_dir is not None and len(speed_samples) > 1:
            save_speed_profile(speed_samples, output_dir, prefix="pull_over")


