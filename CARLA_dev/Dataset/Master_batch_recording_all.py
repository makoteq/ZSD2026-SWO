import carla
import os
import cv2
import numpy as np
import csv
import traceback
from datetime import datetime

# Wyciszenie interfejsu graficznego dla Matplotlib, aby uniknąć błędów wątków
import matplotlib
matplotlib.use('Agg')

import s_single_speeding
import s_pull_over
import s_normal_traffic
import s_overtake

#sensor loc
#todo remove const and add random
SENSOR_X = 181.0
SENSOR_Y = 107.0
SENSOR_Z = 6.0
SENSOR_YAW = 180.0
CAMERA_PITCH = -12.0
RADAR_PITCH = -8.0
#debug lines                #todo find a better way to disp them
DRAW_DEBUG_LINES = True
SIDE_LEFT_Y = 111.0
SIDE_RIGHT_Y = 104.0
SIDE_LINE_X_START = 20.0
SIDE_LINE_X_END = 200.0
SIDE_LINE_Z = 0.3
SIDE_LINE_COLOR = carla.Color(255, 255, 255)
SIDE_LINE_THICKNESS = 0.05

#scenarios------------------------------------------------------------------------------
SCENARIOS_TO_RUN=[
    #standard scenarios:
    ("normal_traffic", s_normal_traffic, "Normal Traffic"),
    ("single_speeding", s_single_speeding, "Single Speeding"),
    #("speeding", scenario_speeding, "Speeding"),
    #("lane_change", scenario_lane_change, "Lane Change"),
    ("overtaking", s_overtake, "Overtaking"),
    #anomalys:
    ("pull_over", s_pull_over, "Pull Over To Shoulder"),
]
SAMPLES_PER_SCENARIO = 3
RECORD_DURATION_SEC = 30.0      #for normal traffic only

#weather -----------------------------------------------------------------------------
WEATHER_CONFIG = {
    "day": carla.WeatherParameters.ClearNoon,
    "rain": carla.WeatherParameters.HardRainNoon,
    "fog": carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=10.0,
        sun_altitude_angle=45.0,
        fog_density=40.0,
        fog_distance=15.0,
        fog_falloff=2.0
    )
}

#radar ---------------------------------------------------------------------------------
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
#camera ---------------------------------------------------------------------------------
CAMERA_TICK = 0.05
VIDEO_FPS = 1.0 / CAMERA_TICK
#global states --------------------------------------------------------------------------
radar_actor = None
video_writer = None
radar_csv_file = None
radar_csv_writer = None
video_path = None
#-----------------------------------------------------------------------------------------

#destroys cars safely, if code tried to destroy a none existent actor carla would break
def safe_destroy_actor(actor):
    if actor is None:
        return
    try:
        actor.stop()
    except Exception:
        pass

    try:
        actor.destroy()
    except Exception:
        pass

def safe_destroy_vehicles_batch(world, client):
    try:
        vehicles = world.get_actors().filter("vehicle.*")
        ids = [a.id for a in vehicles]
        if not ids:
            return

        cmds = [carla.command.DestroyActor(x) for x in ids]
        client.apply_batch_sync(cmds, True)
    except Exception as e:
        print(f"[WARN] safe_destroy_vehicles_batch failed: {e}")

#recording and misc ---------------------------------------------------------------------

def start_recording(run_dir: str, weather_name: str):
    global video_writer, radar_csv_file, radar_csv_writer, video_path

    stop_recording()
    os.makedirs(run_dir, exist_ok=True)

    radar_log_path = os.path.join(run_dir, f"radar_{weather_name}.csv")
    video_path = os.path.join(run_dir, f"video_{weather_name}.mp4")

    radar_csv_file = open(radar_log_path, "w", newline="", encoding="utf-8")
    radar_csv_writer = csv.writer(radar_csv_file)
    radar_csv_writer.writerow([
        "timestamp", "frame",
        "radial_velocity", "azimuth", "altitude", "depth",
        "x_sensor", "y_sensor", "z_sensor",
        "x_world", "y_world", "z_world"
    ])

def stop_recording():
    global video_writer, radar_csv_file, radar_csv_writer, video_path

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

    video_path = None

#callbacks for radar and camera ----------------------------------------------------------------

def radar_callback(sensor_data):
    global radar_actor, radar_csv_writer
    if radar_csv_writer is None or radar_actor is None:
        return

    points = np.frombuffer(
        sensor_data.raw_data,dtype=np.float32).reshape((len(sensor_data), 4))

    vel   = points[:, 0]
    az    = points[:, 1]
    alt   = points[:, 2]
    depth = points[:, 3]

    x = depth * np.cos(alt) * np.sin(az)
    y = depth * np.cos(alt) * np.cos(az)
    z = depth * np.sin(alt)

    transform = radar_actor.get_transform()

    rows = []
    for i in range(len(points)):
        loc       = carla.Location(x=float(x[i]), y=float(y[i]), z=float(z[i]))
        world_loc = transform.transform(loc)
        rows.append([
            float(sensor_data.timestamp), int(sensor_data.frame),
            float(vel[i]), float(az[i]), float(alt[i]), float(depth[i]),
            float(x[i]),  float(y[i]),  float(z[i]),
            float(world_loc.x), float(world_loc.y), float(world_loc.z)
        ])

    radar_csv_writer.writerows(rows)

def camera_callback(image):
    global video_writer, video_path

    if not video_path:
        return

    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = np.reshape(img, (image.height, image.width, 4))
    bgr = img[:, :, :3].copy()

    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        path   = video_path
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

#enviroment and blueprint presets ----------------------------------------------------------------
def draw_debug_side_lines(world):
    if not DRAW_DEBUG_LINES:
        return

    left_a = carla.Location(x=SIDE_LINE_X_START, y=SIDE_LEFT_Y, z=SIDE_LINE_Z)
    left_b = carla.Location(x=SIDE_LINE_X_END, y=SIDE_LEFT_Y, z=SIDE_LINE_Z)
    right_a = carla.Location(x=SIDE_LINE_X_START, y=SIDE_RIGHT_Y, z=SIDE_LINE_Z)
    right_b = carla.Location(x=SIDE_LINE_X_END, y=SIDE_RIGHT_Y, z=SIDE_LINE_Z)

    world.debug.draw_line(left_a, left_b, thickness=SIDE_LINE_THICKNESS, color=SIDE_LINE_COLOR, life_time=0.0, persistent_lines=True)
    world.debug.draw_line(right_a, right_b, thickness=SIDE_LINE_THICKNESS, color=SIDE_LINE_COLOR, life_time=0.0, persistent_lines=True)

def apply_radar_preset(radar_bp, preset_name: str):
    preset = RADAR_PRESETS[preset_name]
    for k, v in preset.items():
        radar_bp.set_attribute(k, v)

def setup_environment(client):
    world = client.load_world('Town02')
    
    labels_to_hide = [
        carla.CityObjectLabel.Buildings,
        carla.CityObjectLabel.Fences,
        carla.CityObjectLabel.Vegetation,
        carla.CityObjectLabel.Other,
        carla.CityObjectLabel.Walls,
        carla.CityObjectLabel.GuardRail
    ]
    all_to_hide = [obj.id for label in labels_to_hide for obj in world.get_environment_objects(label)]
    world.enable_environment_objects(all_to_hide, False)

    draw_debug_side_lines(world)

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        carla.Location(x=SENSOR_X, y=SENSOR_Y, z=SENSOR_Z),
        carla.Rotation(pitch=CAMERA_PITCH, yaw=SENSOR_YAW)
    ))

    blueprint_library = world.get_blueprint_library()

    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("fov", "14.0")
    camera_bp.set_attribute("image_size_y", "1920")
    camera_bp.set_attribute("image_size_x", "1080")
    camera_bp.set_attribute("sensor_tick", str(CAMERA_TICK))
    camera_bp.set_attribute("bloom_intensity", "0.0")
    camera_bp.set_attribute("exposure_mode", "manual")
    camera_bp.set_attribute("exposure_compensation", "0.0")
    camera_bp.enable_postprocess_effects = True

    radar_bp = blueprint_library.find("sensor.other.radar")
    apply_radar_preset(radar_bp, RADAR_PROFILE_NAME)

    return world, blueprint_library, camera_bp, radar_bp

def spawn_sensors_fixed(world, camera_bp, radar_bp):
    global radar_actor

    t_camera = carla.Transform(
        carla.Location(x=SENSOR_X, y=SENSOR_Y, z=SENSOR_Z),
        carla.Rotation(pitch=CAMERA_PITCH, yaw=SENSOR_YAW)
    )
    
    t_radar = carla.Transform(
        carla.Location(x=SENSOR_X, y=SENSOR_Y, z=SENSOR_Z),
        carla.Rotation(pitch=RADAR_PITCH, yaw=SENSOR_YAW)
    )

    camera = world.spawn_actor(camera_bp, t_camera)
    radar = world.spawn_actor(radar_bp, t_radar)

    radar_actor = radar
    radar.listen(safe_callback_wrapper(radar_callback))
    camera.listen(safe_callback_wrapper(camera_callback))

    return camera, radar

#main =====================================================================================

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    try:
        world, blueprint_library, camera_bp, radar_bp = setup_environment(client)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir = f"recordings_{session_id}"

        print("\n" + "=" * 70)
        print(f"BATCH start | scenarios={len(SCENARIOS_TO_RUN)} | samples={SAMPLES_PER_SCENARIO}")
        print(f"Radar={RADAR_PROFILE_NAME} | VIDEO_FPS={VIDEO_FPS:.1f} | CAMERA_TICK={CAMERA_TICK}")
        print("=" * 70)

        for scenario_key, scenario_module, scenario_label in SCENARIOS_TO_RUN:
            for case_idx in range(1, SAMPLES_PER_SCENARIO + 1):
                for weather_name, weather_params in WEATHER_CONFIG.items():
                    
                    run_dir = os.path.join(root_dir, scenario_key, f"case_{case_idx}")
                    print(f"Running: {scenario_label} | Case {case_idx}/{SAMPLES_PER_SCENARIO} | Weather: {weather_name}")

                    try:
                        safe_destroy_vehicles_batch(world, client)
                        world.set_weather(weather_params)
                        
                        start_recording(run_dir, weather_name)
                        camera, radar = spawn_sensors_fixed(world, camera_bp, radar_bp)

                        scenario_module.run(
                            world,
                            blueprint_library,
                            duration_sec=RECORD_DURATION_SEC,
                            output_dir=run_dir
                        )

                    except Exception as e:
                        print(f"[ERROR] {e}")
                        traceback.print_exc()

                    finally:
                        safe_destroy_actor(camera)
                        safe_destroy_actor(radar)
                        stop_recording()

        print("\nZakończono batch.")
        print(f"Wyniki: {root_dir}")

    except Exception as e:
        print(f"\nBłąd środowiska: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()