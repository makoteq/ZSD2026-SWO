import os
import csv
import math
import time
import carla
import random
import matplotlib.pyplot as plt

LANE_A_Y = 109.5
LANE_B_Y = 105.5
START_X = -3.0
SPAWN_Z = 0.8
YAW = 0.0
CROSSING_LINE = 260.0

INITIAL_KPH = 50.0
TARGET_MIN_KPH = 100
TARGET_MAX_KPH = 120

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
#-----------------------------------------------------------------------
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


def run(world, blueprint_library, duration_sec=30.0, output_dir=None):
    destroy_all_vehicles(world)

    v1_model = blueprint_library.find('vehicle.dodge.charger_2020')
    lane_y = random.choice([LANE_A_Y, LANE_B_Y])
    #lane_y = LANE_A_Y


    t1 = carla.Transform(carla.Location(x=START_X, y=lane_y, z=SPAWN_Z), carla.Rotation(yaw=YAW))
    v1 = world.try_spawn_actor(v1_model, t1)
    v1.set_target_velocity(carla.Vector3D(x=INITIAL_KPH / 3.6, y=0.0, z=0.0))

    if v1 is None:
        print("[SPAWN ERROR] single_speeding")
        return

    # initial_speed = random.uniform(40.0, 60.0) / 3.6
    # max_speed = random.uniform(100.0, 150.0) / 3.6
    # boost_trigger_x = random.uniform(40.0, 120.0)

    target_kph = random.uniform(TARGET_MIN_KPH, TARGET_MAX_KPH)
    print(f"[single_speeding] target={target_kph:.1f} km/h")
    speed_samples = []
    t0 = time.time()

    try:
        scenario_active = True
        while scenario_active:
            if not v1.is_alive:
                break
            t_rel = time.time() - t0
            v_kmh = speed_kph(v1)
            speed_samples.append((t_rel, v_kmh))

            loc1 = v1.get_location()
            apply_speed(v1, target_kph, steer=0.0)
            # v1.set_target_velocity(carla.Vector3D(
            #     x=max_speed if loc1.x > boost_trigger_x else initial_speed, y=0.0, z=0.0
            # ))

            if loc1.x > CROSSING_LINE:
                scenario_active = False

            time.sleep(0.05)
    finally:
        destroy_all_vehicles(world)
        if output_dir is not None and len(speed_samples) > 1:
            save_speed_profile(speed_samples, output_dir, prefix="single_speeding")