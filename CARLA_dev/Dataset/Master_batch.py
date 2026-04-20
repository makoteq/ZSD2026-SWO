import carla
import os
import cv2
import numpy as np
import csv
import time
import random
import json
import traceback
from datetime import datetime

import s_single_speeding
import s_pull_over
import s_overtake
import s_normal_trafic

run_start_sim_time = 0.0
# sensor loc
# todo remove const and add random
SENSOR_X = 181.0
SENSOR_Y = 107.0
SENSOR_Z = 6.0
SENSOR_YAW = 180.0
# SENSOR_PITCH = -5.0
# can and radaar params
# ---------------------------------------
CAMERA_Y = 107.0
CAMERA_PITCH = -12.0
CAMERA_FOV = 20.0

RADAR_Y = 107.0
RADAR_PITCH = -8.0
# ---------------------------------------
# debug lines                #todo find a better way to disp them
DRAW_DEBUG_LINES = True
SIDE_LEFT_Y = 111.0
SIDE_RIGHT_Y = 104.0
SIDE_LINE_X_START = 20.0
SIDE_LINE_X_END = 200.0
SIDE_LINE_Z = 0.3
SIDE_LINE_COLOR = carla.Color(255, 255, 255)
SIDE_LINE_THICKNESS = 0.05

# scenarios------------------------------------------------------------------------------
SCENARIOS_TO_RUN = [
    # standard scenarios:
    # ("normal_traffic", s_normal_trafic, "Normal Traffic"),
    # ("single_speeding", s_single_speeding, "Single Speeding"),
    # ("speeding", scenario_speeding, "Speeding"),
    # ("lane_change", scenario_lane_change, "Lane Change"),
    ("overtaking", s_overtake, "Overtaking"),
    # anomalys:
    # ("pull_over", s_pull_over, "Pull Over To Shoulder"),
]
SAMPLES_PER_SCENARIO = 5
RECORD_DURATION_SEC = 30.0  # for normal traffic only
# weather -----------------------------------------------------------------------------
WEATHER_PRESETS = {
    '1': ('ClearNoon', carla.WeatherParameters.ClearNoon),
    '2': ('ClearSunset', carla.WeatherParameters.ClearSunset),
    '3': ('CloudyNoon', carla.WeatherParameters.CloudyNoon),
    '4': ('CloudySunset', carla.WeatherParameters.CloudySunset),
    '5': ('WetNoon', carla.WeatherParameters.WetNoon),
    '6': ('WetSunset', carla.WeatherParameters.WetSunset),
    '7': ('MidRainyNoon', carla.WeatherParameters.MidRainyNoon),
    '8': ('MidRainSunset', carla.WeatherParameters.MidRainSunset),
    '9': ('HardRainNoon', carla.WeatherParameters.HardRainNoon),
    '10': ('HardRainSunset', carla.WeatherParameters.HardRainSunset),
    '11': ('SoftRainNoon', carla.WeatherParameters.SoftRainNoon),
    '12': ('SoftRainSunset', carla.WeatherParameters.SoftRainSunset),
    '13': ('ClearNight', carla.WeatherParameters.ClearNight),
    '14': ('CloudyNight', carla.WeatherParameters.CloudyNight),
    '15': ('WetNight', carla.WeatherParameters.WetNight),
    '16': ('SoftRainNight', carla.WeatherParameters.SoftRainNight),
    '17': ('MidRainyNight', carla.WeatherParameters.MidRainyNight),
    '18': ('HardRainNight', carla.WeatherParameters.HardRainNight),
}
WEATHER_TAGS = {
    "day": "1",
    "night": "13",
    "rain": "9",  # todo fix and shorten the weather situation
    "fog": "3",
}

current_weather_name = 'ClearNoon'
WEATHER_PLAN = ["day"] * SAMPLES_PER_SCENARIO

# radar ---------------------------------------------------------------------------------
RADAR_PROFILE_NAME = "TRUGRD_LR_like"
RADAR_PRESETS = {
    "TRUGRD_LR_like": {
        "horizontal_fov": "18.0",
        "vertical_fov": "6.0",
        "range": "500.0",
        "points_per_second": "14000",
        "sensor_tick": "0.05",
    }
}
# camera ---------------------------------------------------------------------------------
CAMERA_TICK = 0.05
VIDEO_FPS = 1.0 / CAMERA_TICK
# global states --------------------------------------------------------------------------
radar_actor = None
video_writer = None
radar_csv_file = None
radar_csv_writer = None
output_dir = None
video_path = None
radar_log_path = None
session_log_path = None


# -----------------------------------------------------------------------------------------

# destroys cars safely, if code tried to destroy a none existent actor carla would break
def safe_destroy_actor(actor):
    if actor is None:
        return
    try:
        actor_id = actor.id
    except Exception:
        pass

    try:
        actor.stop()
    except Exception:
        pass

    try:
        actor.destroy()
    except RuntimeError as err:
        msg = str(err).lower()
        if ("not found" in msg) or ("already destroyed" in msg):
            return
    except Exception:
        pass


def safe_destroy_vehicles_batch(world, client):
    try:
        vehicles = world.get_actors().filter("vehicle.*")
        ids = [a.id for a in vehicles]
        if not ids:
            return

        cmds = [carla.command.DestroyActor(x) for x in ids]
        responses = client.apply_batch_sync(cmds, True)

        for r in responses:
            if r.error:
                err = str(r.error).lower()
                if "not found" in err or "already destroyed" in err:
                    continue
                print(f"[WARN] batch destroy error: {r.error}")
    except Exception as e:
        print(f"[WARN] safe_destroy_vehicles_batch failed: {e}")


# recording and misc ---------------------------------------------------------------------

def weather_for_run(global_run_idx: int):
    tag = WEATHER_PLAN[(global_run_idx - 1) % len(WEATHER_PLAN)]
    preset_key = WEATHER_TAGS.get(tag, "1")
    name, preset = WEATHER_PRESETS[preset_key]
    return tag, name, preset


def start_recording(run_dir: str):
    global video_writer, radar_csv_file, radar_csv_writer
    global output_dir, video_path, radar_log_path

    stop_recording()

    output_dir = run_dir
    os.makedirs(output_dir, exist_ok=True)

    radar_log_path = os.path.join(output_dir, "radar_points_world.csv")
    video_path = os.path.join(output_dir, "rgb.mp4")

    radar_csv_file = open(radar_log_path, "w", newline="", encoding="utf-8")
    radar_csv_writer = csv.writer(radar_csv_file)
    radar_csv_writer.writerow([
        "timestamp", "frame",
        "radial_velocity", "azimuth", "altitude", "depth",
        "x_sensor", "y_sensor", "z_sensor",
        "x_world", "y_world", "z_world"
    ])

    video_writer = None


def stop_recording():
    global video_writer, radar_csv_file, radar_csv_writer
    global output_dir, video_path, radar_log_path

    if radar_csv_file is not None:
        try:
            radar_csv_file.close()
        except Exception:
            pass
        radar_csv_file = None
        radar_csv_writer = None

    if video_writer is not None:
        try:
            video_writer.release()
        except Exception:
            pass
        video_writer = None

    output_dir = None
    video_path = None
    radar_log_path = None


# callbacks for radar and camera ----------------------------------------------------------------

def radar_callback(sensor_data):
    global radar_actor, radar_csv_writer, run_start_sim_time
    if radar_csv_writer is None or radar_actor is None:
        return

    ts_abs = float(sensor_data.timestamp)
    ts_run = ts_abs - run_start_sim_time if run_start_sim_time is not None else 0.0

    points = np.frombuffer(sensor_data.raw_data, dtype=np.float32).reshape((len(sensor_data), 4))
    vel = points[:, 0]
    az = points[:, 1]
    alt = points[:, 2]
    depth = points[:, 3]

    x = depth * np.cos(alt) * np.sin(az)
    y = depth * np.cos(alt) * np.cos(az)
    z = depth * np.sin(alt)

    transform = radar_actor.get_transform()

    rows = []
    for i in range(len(points)):
        loc = carla.Location(x=float(x[i]), y=float(y[i]), z=float(z[i]))
        world_loc = transform.transform(loc)
        rows.append([
            ts_run, int(sensor_data.frame),
            float(vel[i]), float(az[i]), float(alt[i]), float(depth[i]),
            float(x[i]), float(y[i]), float(z[i]),
            float(world_loc.x), float(world_loc.y), float(world_loc.z)
        ])

    try:
        radar_csv_writer.writerows(rows)
    except ValueError:
        return


def camera_callback(image):
    global video_writer, video_path

    if not video_path:
        return

    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = np.reshape(img, (image.height, image.width, 4))
    bgr = img[:, :, :3].copy()

    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        path = video_path
        if not path:
            return
        video_writer = cv2.VideoWriter(path, fourcc, VIDEO_FPS, (image.width, image.height))
        if not video_writer.isOpened():
            video_writer = None
            return

    video_writer.write(bgr)


def safe_callback_wrapper(fn):
    def _wrapped(data):
        try:
            fn(data)
        except Exception as e:
            print(f"[WARN] callback exception: {e}")
            traceback.print_exc()

    return _wrapped


# enviroment and blueprint presets ----------------------------------------------------------------
def draw_debug_side_lines(world):
    if not DRAW_DEBUG_LINES:
        return

    left_a = carla.Location(x=SIDE_LINE_X_START, y=SIDE_LEFT_Y, z=SIDE_LINE_Z)
    left_b = carla.Location(x=SIDE_LINE_X_END, y=SIDE_LEFT_Y, z=SIDE_LINE_Z)
    right_a = carla.Location(x=SIDE_LINE_X_START, y=SIDE_RIGHT_Y, z=SIDE_LINE_Z)
    right_b = carla.Location(x=SIDE_LINE_X_END, y=SIDE_RIGHT_Y, z=SIDE_LINE_Z)

    world.debug.draw_line(left_a, left_b, thickness=SIDE_LINE_THICKNESS, color=SIDE_LINE_COLOR, life_time=0.0,
                          persistent_lines=True)
    world.debug.draw_line(right_a, right_b, thickness=SIDE_LINE_THICKNESS, color=SIDE_LINE_COLOR, life_time=0.0,
                          persistent_lines=True)


def apply_radar_preset(radar_bp, preset_name: str):
    preset = RADAR_PRESETS[preset_name]
    for k, v in preset.items():
        radar_bp.set_attribute(k, v)


def setup_environment(client):
    world = client.load_world('Town02')
    world.set_weather(carla.WeatherParameters.CloudySunset)

    labels_to_hide = [
        carla.CityObjectLabel.Buildings,
        carla.CityObjectLabel.Fences,
        carla.CityObjectLabel.Vegetation,
        carla.CityObjectLabel.Other,
        carla.CityObjectLabel.Walls,
        # carla.CityObjectLabel.Poles,
        carla.CityObjectLabel.GuardRail
    ]
    all_to_hide = [obj.id for label in labels_to_hide for obj in world.get_environment_objects(label)]
    world.enable_environment_objects(all_to_hide, False)

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        carla.Location(x=SENSOR_X, y=SENSOR_Y, z=SENSOR_Z),
        carla.Rotation(pitch=CAMERA_PITCH, yaw=SENSOR_YAW)
    ))

    draw_debug_side_lines(world)

    blueprint_library = world.get_blueprint_library()

    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("fov", str(CAMERA_FOV))
    camera_bp.set_attribute("image_size_y", "1080")
    camera_bp.set_attribute("image_size_x", "1920")
    camera_bp.set_attribute("sensor_tick", str(CAMERA_TICK))
    camera_bp.set_attribute("bloom_intensity", "0.0")
    camera_bp.set_attribute("exposure_mode", "manual")
    camera_bp.set_attribute("exposure_compensation", "0.0")
    camera_bp.enable_postprocess_effects = True

    radar_bp = blueprint_library.find("sensor.other.radar")
    apply_radar_preset(radar_bp, RADAR_PROFILE_NAME)

    return world, blueprint_library, camera_bp, radar_bp


def spawn_sensors_fixed(world, camera_bp, radar_bp):  # todo remove const and add random
    global radar_actor

    cam = carla.Transform(
        carla.Location(x=SENSOR_X, y=CAMERA_Y, z=SENSOR_Z),
        carla.Rotation(pitch=CAMERA_PITCH, yaw=SENSOR_YAW)
    )
    rad = carla.Transform(
        carla.Location(x=SENSOR_X, y=RADAR_Y, z=SENSOR_Z),
        carla.Rotation(pitch=RADAR_PITCH, yaw=SENSOR_YAW)
    )

    camera = world.spawn_actor(camera_bp, cam)
    radar = world.spawn_actor(radar_bp, rad)

    radar_actor = radar
    radar.listen(safe_callback_wrapper(radar_callback))
    camera.listen(safe_callback_wrapper(camera_callback))

    return camera, radar, cam, rad


# logs -----------------------------------------------------------------------------------

# for saving sensor config to find best param
def save_sensor_config(run_dir, camera_bp, radar_bp, camera_transform, radar_transform):
    manifest_path = os.path.join(run_dir, "sensor_config.json")

    data = {
        "radar": {
            "horizontal_fov": radar_bp.get_attribute("horizontal_fov").as_float(),
            "vertical_fov": radar_bp.get_attribute("vertical_fov").as_float(),
            "pitch_deg": float(radar_transform.rotation.pitch),
        },
        "camera": {
            "fov": camera_bp.get_attribute("fov").as_float(),
            "image_size_x": camera_bp.get_attribute("image_size_x").as_int(),
            "image_size_y": camera_bp.get_attribute("image_size_y").as_int(),
            "pitch_deg": float(camera_transform.rotation.pitch),
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[CFG] saved: {manifest_path}")


def init_batch_output():
    global session_log_path

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = f"batch_output_{session_id}"
    os.makedirs(root_dir, exist_ok=True)

    session_log_path = os.path.join(root_dir, "run_log.csv")
    with open(session_log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_timestamp",
            "global_run_index",
            "scenario_key",
            "scenario_label",
            "sample_index",
            "run_dir",
            "weather_tag",
            "weather_name",
            "radar_profile",
            "sensor_x",
            "sensor_y",
            "sensor_z",
            "sensor_yaw",
            "duration_sec",
            "status",
            "error"
        ])
    return root_dir


def append_run_log(row):
    with open(session_log_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# main =====================================================================================

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    camera = None
    radar = None

    try:
        world, blueprint_library, camera_bp, radar_bp = setup_environment(client)
        root_dir = init_batch_output()

        total_runs = len(SCENARIOS_TO_RUN) * SAMPLES_PER_SCENARIO
        global_idx = 0

        print("\n" + "=" * 70)
        print(f"BATCH start | scenarios={len(SCENARIOS_TO_RUN)} | samples={SAMPLES_PER_SCENARIO} | total={total_runs}")
        print(f"Radar={RADAR_PROFILE_NAME} | VIDEO_FPS={VIDEO_FPS:.1f} | CAMERA_TICK={CAMERA_TICK}")
        print("=" * 70)

        for scenario_key, scenario_module, scenario_label in SCENARIOS_TO_RUN:
            for sample_idx in range(1, SAMPLES_PER_SCENARIO + 1):
                global_idx += 1
                run_dir = os.path.join(root_dir, scenario_key, f"sample_{sample_idx:03d}")
                os.makedirs(run_dir, exist_ok=True)

                status = "ok"
                error_msg = ""
                t0 = time.time()

                print(f"\n[{global_idx}/{total_runs}] {scenario_label} | sample={sample_idx}")

                cam = None
                rad = None

                try:
                    global run_start_sim_time
                    for _ in range(2):
                        world.tick()
                    snap = world.get_snapshot()
                    run_start_sim_time = float(snap.timestamp.elapsed_seconds)

                    safe_destroy_vehicles_batch(world, client)

                    camera = None
                    radar = None

                    run_start_sim_time = None
                    stop_recording()
                    start_recording(run_dir)

                    weather_tag, weather_name, weather_obj = weather_for_run(global_idx)
                    world.set_weather(weather_obj)

                    for _ in range(3):
                        world.tick()

                    camera, radar, cam, rad = spawn_sensors_fixed(world, camera_bp, radar_bp)
                    save_sensor_config(run_dir, camera_bp, radar_bp, cam, rad)

                    for _ in range(2):
                        world.tick()
                    snap = world.get_snapshot()
                    run_start_sim_time = float(snap.timestamp.elapsed_seconds)

                    scenario_module.run(
                        world,
                        blueprint_library,
                        duration_sec=RECORD_DURATION_SEC,
                        output_dir=run_dir
                    )

                    time.sleep(0.3)

                except Exception as e:
                    status = "error"
                    error_msg = str(e)
                    print(f"[ERROR] {e}")
                    traceback.print_exc()

                finally:
                    run_start_sim_time = None
                    safe_destroy_actor(camera)
                    safe_destroy_actor(radar)
                    camera = None
                    radar = None

                    safe_destroy_vehicles_batch(world, client)
                    stop_recording()

                    append_run_log([
                        datetime.now().isoformat(timespec="seconds"),
                        global_idx,
                        scenario_key,
                        scenario_label,
                        sample_idx,
                        run_dir,
                        weather_tag,
                        weather_name,
                        RADAR_PROFILE_NAME,
                        f"{SENSOR_X:.3f}",
                        f"{SENSOR_Y:.3f}",
                        f"{SENSOR_Z:.3f}",
                        f"{SENSOR_YAW:.1f}",
                        f"{RECORD_DURATION_SEC:.1f}",
                        status,
                        error_msg
                    ])

        print("\nZakończono batch.")
        print(f"Wyniki: {root_dir}")
        print(f"Log:    {session_log_path}")

    except Exception as e:
        print(f"\nBłąd środowiska: {e}")
        traceback.print_exc()

    finally:
        safe_destroy_actor(camera)
        safe_destroy_actor(radar)
        stop_recording()


if __name__ == "__main__":
    main()
