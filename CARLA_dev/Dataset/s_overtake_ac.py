import carla
import time
import random
import math
import os
import csv
import numpy as np

# Lane parameters
LANE_A_Y = 109.5
LANE_B_Y = 105.5
START_X = -3.0
SPAWN_Z = 0.8
YAW = 0.0
CROSSING_LINE = 206.0
DT = 0.05

# Car speed parameters
CRUISE_MIN_KPH = 50.0
CRUISE_MAX_KPH = 50.0
OVERTAKE_MAX_KPH = 70.0

OVERTAKE_START_MIN = 110.0
OVERTAKE_START_MAX = 120.0

PASS_GAP_X = 10
CUTIN_Y_TOL = 0.10

# Utils
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

    # Apply variables to carla car
    vehicle.apply_control(carla.VehicleControl(
        throttle=throttle,
        brake=brake,
        steer=steer,
        hand_brake=False,
        manual_gear_shift=False
    ))

def steer_to_y(y_target, y_current, y_err_prev,  kp=0.1, kd=0.25, limit=0.14):
    y_err = y_target - y_current
    d_err = y_err - y_err_prev
    steer = kp * y_err + kd * d_err
    steer = max(-limit, min(limit, steer))
    return steer, y_err

# Main

def run(world, blueprint_library, duration_sec=30.0, output_dir=None):
    destroy_all_vehicles(world)

    bp_fast = blueprint_library.find('vehicle.dodge.charger_2020')
    bp_slow = blueprint_library.find('vehicle.dodge.charger_2020')

    y_slow = LANE_A_Y
    y_fast = LANE_B_Y

    t_slow = carla.Transform(carla.Location(x=START_X+10, y=y_slow, z=SPAWN_Z), carla.Rotation(yaw=YAW))
    t_fast = carla.Transform(carla.Location(x=START_X, y=y_fast, z=SPAWN_Z), carla.Rotation(yaw=YAW))

    v_slow = world.try_spawn_actor(bp_slow, t_slow)
    v_fast = world.try_spawn_actor(bp_fast, t_fast)

    if v_slow is None or v_fast is None:
        print("[SPAWN ERROR] realistic_overtake")
        destroy_all_vehicles(world)
        return

    slow_cruise = random.uniform(CRUISE_MIN_KPH, CRUISE_MAX_KPH)
    fast_cruise = max(slow_cruise - random.uniform(0.0, 3.0), 38.0)
    overtake_start_x = START_X + random.uniform(OVERTAKE_START_MIN, OVERTAKE_START_MAX)
    print(f"[overtake] start_x={overtake_start_x:.1f}")
    overtake_target = OVERTAKE_MAX_KPH

    print(f"[overtake] slow={slow_cruise:.1f} fast_start={fast_cruise:.1f} overtake={overtake_target:.1f}")

    t0 = time.time()
    speed_fast = []
    speed_slow = []

    phase = "cruise"
    y_err_prev = 0.0

    try:
        while True:
            if (not v_fast.is_alive) or (not v_slow.is_alive):
                break

            now = time.time()
            t_rel = now - t0

            loc_fast = v_fast.get_location()
            loc_slow = v_slow.get_location()

            speed_fast.append((t_rel, speed_kph(v_fast)))
            speed_slow.append((t_rel, speed_kph(v_slow)))

            if duration_sec is not None and t_rel >= duration_sec:
                break
            if loc_fast.x > CROSSING_LINE or loc_slow.x > CROSSING_LINE:
                break

            if phase == "cruise":
                apply_speed(v_fast, slow_cruise, steer=0.0)
                apply_speed(v_slow, slow_cruise, steer=0.0)
                if loc_slow.x >= overtake_start_x:
                    phase = "boost"
            elif phase == "boost":
                apply_speed(v_fast, overtake_target, steer=0.0)
                apply_speed(v_slow, slow_cruise, steer=0.0)
                if (loc_fast.x - loc_slow.x) >= PASS_GAP_X:
                    phase = "cut_in"
                    y_err_prev = y_slow - loc_fast.y
            elif phase == "cut_in":
                steer_cmd, y_err_prev = steer_to_y(
                    y_target=y_slow,
                    y_current=loc_fast.y,
                    y_err_prev=y_err_prev,
                    kp=0.03, kd=0.55, limit=0.11
                )
                apply_speed(v_fast, overtake_target, steer=steer_cmd)
                apply_speed(v_slow, slow_cruise, steer=0.0)
                if abs(y_slow - loc_fast.y) < CUTIN_Y_TOL:
                    v_fast.apply_control(carla.VehicleControl(
                        throttle=0.12, brake=0.0, steer=0.0,
                        hand_brake=False, manual_gear_shift=False
                    ))
                    phase = "stabilize"
            elif phase == "stabilize":
                steer_cmd, y_err_prev = steer_to_y(
                    y_target=y_slow,
                    y_current=loc_fast.y,
                    y_err_prev=y_err_prev,
                    kp=0.05, kd=0.30, limit=0.07
                )
                apply_speed(v_fast, min(50.0, overtake_target - 15.0), steer=steer_cmd)
                apply_speed(v_slow, slow_cruise, steer=0.0)
            time.sleep(DT)

    finally:
        destroy_all_vehicles(world)