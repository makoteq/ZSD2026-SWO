import carla
import os
import cv2
import numpy as np
import csv
import time
import random
from datetime import datetime

import scenario_speeding
import scenario_single_speeding
import scenario_lane_change
import scenario_overtaking
import scenario_normal_traffic

VIDEO_FPS = 60.0

radar_actor       = None
video_writer      = None
video_writer_rfov = None
radar_csv_file    = None
radar_csv_writer  = None
output_dir        = None
video_path        = None
video_path_rfov   = None
radar_log_path    = None


session_log_path  = None

WEATHER_PRESETS = {
    '1':  ('ClearNoon',            carla.WeatherParameters.ClearNoon),
    '2':  ('ClearSunset',          carla.WeatherParameters.ClearSunset),
    '3':  ('CloudyNoon',           carla.WeatherParameters.CloudyNoon),
    '4':  ('CloudySunset',         carla.WeatherParameters.CloudySunset),
    '5':  ('WetNoon',              carla.WeatherParameters.WetNoon),
    '6':  ('WetSunset',            carla.WeatherParameters.WetSunset),
    '7':  ('MidRainyNoon',         carla.WeatherParameters.MidRainyNoon),
    '8':  ('MidRainSunset',        carla.WeatherParameters.MidRainSunset),
    '9':  ('HardRainNoon',         carla.WeatherParameters.HardRainNoon),
    '10': ('HardRainSunset',       carla.WeatherParameters.HardRainSunset),
    '11': ('SoftRainNoon',         carla.WeatherParameters.SoftRainNoon),
    '12': ('SoftRainSunset',       carla.WeatherParameters.SoftRainSunset),
    '13': ('ClearNight',           carla.WeatherParameters.ClearNight),
    '14': ('CloudyNight',          carla.WeatherParameters.CloudyNight),
    '15': ('WetNight',             carla.WeatherParameters.WetNight),
    '16': ('SoftRainNight',        carla.WeatherParameters.SoftRainNight),
    '17': ('MidRainyNight',        carla.WeatherParameters.MidRainyNight),
    '18': ('HardRainNight',        carla.WeatherParameters.HardRainNight),
}

current_weather_name = 'ClearNoon'


RADAR_PROFILE_NAME = "LRR_12deg_200m_SIM"

#plan pogody dla każdego uruchomienia
WEATHER_PLAN = [
    "day", "day", "day", "day", "day",
    "night", "night", "night", "night", "night",
    "rain", "rain", "rain", "rain", "rain",
    "fog", "fog", "fog", "fog", "fog",
    "rain_night", "rain_night", "rain_night", "rain_night", "rain_night",
    "fog_night", "fog_night", "fog_night", "fog_night", "fog_night",
]


def start_recording(run_dir: str):
    global video_writer, video_writer_rfov, radar_csv_file, radar_csv_writer
    global output_dir, video_path, video_path_rfov, radar_log_path

    stop_recording()

    #zmiana nazwy scenariuszy
    output_dir = run_dir
    os.makedirs(output_dir, exist_ok=True)

    radar_log_path  = os.path.join(output_dir, "radar_points_world.csv")
    video_path      = os.path.join(output_dir, "rgb.mp4")
    video_path_rfov = os.path.join(output_dir, "radar_FOV.mp4")

    radar_csv_file   = open(radar_log_path, "w", newline="", encoding="utf-8")
    radar_csv_writer = csv.writer(radar_csv_file)
    radar_csv_writer.writerow([
        "timestamp", "frame",
        "radial_velocity", "azimuth", "altitude", "depth",
        "x_sensor", "y_sensor", "z_sensor",
        "x_world",  "y_world",  "z_world"
    ])

    video_writer      = None
    video_writer_rfov = None


def stop_recording():
    global video_writer, video_writer_rfov, radar_csv_file, radar_csv_writer
    global output_dir, video_path, video_path_rfov, radar_log_path

    if radar_csv_file is not None:
        radar_csv_file.close()
        radar_csv_file   = None
        radar_csv_writer = None
        print(f"  Radar data saved in: {radar_log_path}")

    if video_writer is not None:
        video_writer.release()
        video_writer = None
        print(f"  Video saved in: {video_path}")

    if video_writer_rfov is not None:
        video_writer_rfov.release()
        video_writer_rfov = None
        print(f"  Radar-FOV visualization saved in: {video_path_rfov}")

    output_dir      = None
    video_path      = None
    video_path_rfov = None
    radar_log_path  = None


def radar_callback(sensor_data):
    global radar_actor, radar_csv_writer

    if radar_csv_writer is None or radar_actor is None:
        return

    points = np.frombuffer(sensor_data.raw_data, dtype=np.float32)
    points = np.reshape(points, (len(sensor_data), 4))

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


def camera_rfov_callback(image):
    global video_writer_rfov, video_path_rfov

    if not video_path_rfov:
        return

    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = np.reshape(img, (image.height, image.width, 4))
    bgr = img[:, :, :3].copy()

    if video_writer_rfov is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        path   = video_path_rfov
        if not path:
            return
        video_writer_rfov = cv2.VideoWriter(path, fourcc, VIDEO_FPS, (image.width, image.height))
        if not video_writer_rfov.isOpened():
            video_writer_rfov = None
            return

    video_writer_rfov.write(bgr)


#cleanup po każdym scenariuszu
def destroy_all_vehicles(world):
    for a in world.get_actors().filter('vehicle.*'):
        if a.is_alive:
            try:
                a.destroy()
            except Exception:
                pass


#dla bezpieczeństwa
def destroy_actor(actor):
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


def change_weather(world):
    global current_weather_name

    print("Change weather")
    for key, (name, _) in WEATHER_PRESETS.items():
        marker = " <-" if name == current_weather_name else ""
        print(f"  {key:>2}. {name}{marker}")
    print("   0. Anuluj")

    choice = input("\nset weather: ").strip()

    if choice == '0':
        return

    if choice not in WEATHER_PRESETS:
        print("no such preset.")
        return

    name, preset = WEATHER_PRESETS[choice]
    world.set_weather(preset)
    current_weather_name = name
    print(f"  Weather set to {name}")


def weather_by_name(name: str):
    for _, (n, preset) in WEATHER_PRESETS.items():
        if n == name:
            return n, preset
    return "ClearNoon", carla.WeatherParameters.ClearNoon


#preset lekkiej mgły ze słońcem
def build_fog_from(base: carla.WeatherParameters, fog_density=85.0, fog_distance=15.0, fog_falloff=1.0):
    return carla.WeatherParameters(
        cloudiness=base.cloudiness,
        precipitation=base.precipitation,
        precipitation_deposits=base.precipitation_deposits,
        wind_intensity=base.wind_intensity,
        sun_azimuth_angle=base.sun_azimuth_angle,
        sun_altitude_angle=base.sun_altitude_angle,
        fog_density=fog_density,
        fog_distance=fog_distance,
        wetness=base.wetness,
        fog_falloff=fog_falloff,
        scattering_intensity=base.scattering_intensity,
        mie_scattering_scale=base.mie_scattering_scale,
        rayleigh_scattering_scale=base.rayleigh_scattering_scale
    )



def weather_from_tag(tag: str):
    if tag == "day":
        n, p = weather_by_name("ClearNoon")
        return n, p, tag
    if tag == "night":
        n, p = weather_by_name("ClearNight")
        return n, p, tag
    if tag == "rain":
        n, p = weather_by_name("HardRainNoon")
        return n, p, tag
    if tag == "fog":
        n, base = weather_by_name("CloudyNoon")
        return n, build_fog_from(base, fog_density=85.0, fog_distance=15.0, fog_falloff=1.0), tag
    if tag == "rain_night":
        n, p = weather_by_name("HardRainNight")
        return n, p, tag
    if tag == "fog_night":
        n, base = weather_by_name("CloudyNight")
        return n, build_fog_from(base, fog_density=80.0, fog_distance=18.0, fog_falloff=1.0), tag

    n, p = weather_by_name("ClearNoon")
    return n, p, "default"


#wybór pogody wg funkcji weather_from_tag
def weather_for_run(run_idx: int):
    tag = WEATHER_PLAN[run_idx - 1]  # 1..30
    return weather_from_tag(tag)


def setup_environment(client):
    global radar_actor

    print("Ładowanie mapy i ustawianie środowiska...")

    world = client.load_world('Town01')
    world.set_weather(carla.WeatherParameters.ClearNoon)

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        carla.Location(x=400.0, y=200.0, z=6.0),
        carla.Rotation(pitch=-10.0, yaw=-105.0)
    ))

    blueprint_library = world.get_blueprint_library()
    #kamera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('fov', '20.0')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('sensor_tick', '0.05')
    camera_bp.set_attribute('bloom_intensity', '0.0')
    camera_bp.set_attribute('exposure_mode', 'manual')
    camera_bp.set_attribute('exposure_compensation', '0.0')
    camera_bp.enable_postprocess_effects = True
    #kamera
    camera_rfov_bp = blueprint_library.find('sensor.camera.rgb')
    camera_rfov_bp.set_attribute('fov', '15')
    camera_rfov_bp.set_attribute('image_size_x', '1920')
    camera_rfov_bp.set_attribute('image_size_y', '1280')
    camera_rfov_bp.set_attribute('sensor_tick', '0.05')
    #radar
    radar_bp = blueprint_library.find('sensor.other.radar')
    radar_bp.set_attribute('horizontal_fov', '15.0')
    radar_bp.set_attribute('vertical_fov', '10.0')
    radar_bp.set_attribute('range', '200.0')
    radar_bp.set_attribute('points_per_second', '3000')
    radar_bp.set_attribute('sensor_tick', '0.1')

    return world, blueprint_library, camera_bp, camera_rfov_bp, radar_bp


#losowanie pozycji sensorów
def spawn_sensors_for_run(world, camera_bp, camera_rfov_bp, radar_bp):
    global radar_actor

    camera_y = random.uniform(-174.0, -168.0)
    radar_y  = camera_y + random.uniform(-1, 1)

    camera_transform = carla.Transform(
        carla.Location(x=190.0, y=camera_y, z=6.0),
        carla.Rotation(pitch=-5.0, yaw=180.0)
    )

    radar_transform = carla.Transform(
        carla.Location(x=190.0, y=radar_y, z=6.0),
        carla.Rotation(pitch=-5.0, yaw=180.0)
    )

    camera = world.spawn_actor(camera_bp, camera_transform)
    camera_rfov = world.spawn_actor(camera_rfov_bp, radar_transform)
    radar = world.spawn_actor(radar_bp, radar_transform)

    radar_actor = radar

    radar.listen(lambda data: radar_callback(data))
    camera.listen(lambda data: camera_callback(data))
    camera_rfov.listen(lambda data: camera_rfov_callback(data))

    return camera, camera_rfov, radar, camera_y, radar_y


def init_batch_output():
    global session_log_path

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = f"scenario5_normal_traffic_output_{session_id}"
    os.makedirs(root_dir, exist_ok=True)

    session_log_path = os.path.join(root_dir, "run_log.csv")
    with open(session_log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_timestamp",
            "run_index",
            "run_dir",
            "camera_y",
            "radar_y",
            "weather_tag",
            "weather_name",
            "radar_profile",
            "status",
            "error",
            "duration_sec"
        ])
    return root_dir



def append_run_log(row):
    with open(session_log_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    camera = None
    camera_rfov = None
    radar = None

    try:
        world, blueprint_library, camera_bp, camera_rfov_bp, radar_bp = setup_environment(client)

        root_dir = init_batch_output()
        total_runs = len(WEATHER_PLAN)

        print("\n" + "=" * 50)
        print("   BATCH: NORMALNY RUCH (30 RUND)")  #todo zautomatyzować zapis i batch pogody
        print("=" * 50)

        for run_idx in range(1, total_runs + 1):
            run_name = f"run_{run_idx:03d}"
            run_dir = os.path.join(root_dir, run_name)

            status = "ok"
            error_msg = ""
            weather_tag = "default"
            weather_name = current_weather_name
            cam_y = None
            rad_y = None
            t0 = time.time()

            print(f"\n[{run_idx}/{total_runs}] Start {run_name}")

            try:
                destroy_all_vehicles(world)
                stop_recording()
                start_recording(run_dir)

                weather_name, weather_obj, weather_tag = weather_for_run(run_idx)
                world.set_weather(weather_obj)

                for _ in range(3):
                    world.tick()

                camera, camera_rfov, radar, cam_y, rad_y = spawn_sensors_for_run(
                    world, camera_bp, camera_rfov_bp, radar_bp
                )

                print(
                    f"  weather={weather_name} ({weather_tag}), "
                    f"camera_y={cam_y:.3f}, radar_y={rad_y:.3f}"
                )

                scenario_normal_traffic.run(world, blueprint_library, duration_sec=30.0)

                time.sleep(0.3)

            except Exception as e:
                status = "error"
                error_msg = str(e)
                print(f"  Błąd podczas runu: {e}")

            finally:
                destroy_actor(camera)
                destroy_actor(camera_rfov)
                destroy_actor(radar)

                camera = None
                camera_rfov = None
                radar = None

                destroy_all_vehicles(world)
                stop_recording()

                dt = round(time.time() - t0, 3)
                append_run_log([
                    datetime.now().isoformat(timespec="seconds"),
                    run_idx,
                    run_dir,
                    "" if cam_y is None else f"{cam_y:.6f}",
                    "" if rad_y is None else f"{rad_y:.6f}",
                    weather_tag,
                    weather_name,
                    RADAR_PROFILE_NAME,
                    status,
                    error_msg,
                    dt
                ])

        print("\nZakończono batch.")
        print(f"Wyniki: {root_dir}")
        print(f"Log:    {session_log_path}")

    except Exception as e:
        print(f"\nBłąd środowiska: {e}")

    finally:
        stop_recording()
        try:
            destroy_actor(camera)
            destroy_actor(camera_rfov)
            destroy_actor(radar)
        except Exception:
            pass


if __name__ == '__main__':
    main()