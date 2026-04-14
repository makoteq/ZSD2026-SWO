import carla
import os
import cv2
import numpy as np
import csv
import time
from datetime import datetime

import scenario_speeding
import scenario_single_speeding
import scenario_lane_change
import scenario_overtaking

VIDEO_FPS = 20.0

radar_actor       = None
video_writer      = None
video_writer_rfov = None
radar_csv_file    = None
radar_csv_writer  = None
output_dir        = None
video_path        = None
video_path_rfov   = None
radar_log_path    = None

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


def start_recording(scenario_name: str):
    global video_writer, video_writer_rfov, radar_csv_file, radar_csv_writer
    global output_dir, video_path, video_path_rfov, radar_log_path

    stop_recording()

    session_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir      = f"{scenario_name}_output_{session_id}"
    os.makedirs(output_dir, exist_ok=True)

    radar_log_path  = os.path.join(output_dir, "radar_points_world.csv")
    video_path      = os.path.join(output_dir, "rgb.mp4")
    video_path_rfov = os.path.join(output_dir, "radar_FOV.mp4")

    radar_csv_file   = open(radar_log_path, "w", newline="")
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

    if radar_csv_writer is None:
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
            sensor_data.timestamp, sensor_data.frame,
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


def run_empty_road(world):
    for i in range(60, 0, -5):
        print(f"  {i}s left")
        time.sleep(5)


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

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('fov', '20.0')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('sensor_tick', '0.05')
    camera_bp.set_attribute('bloom_intensity', '0.0')
    camera_bp.set_attribute('exposure_mode', 'manual')
    camera_bp.set_attribute('exposure_compensation', '0.0')
    camera_bp.enable_postprocess_effects = False

    camera_rfov_bp = blueprint_library.find('sensor.camera.rgb')
    camera_rfov_bp.set_attribute('fov', '15')
    camera_rfov_bp.set_attribute('image_size_x', '1920')
    camera_rfov_bp.set_attribute('image_size_y', '1280')
    camera_rfov_bp.set_attribute('sensor_tick', '0.05')

    radar_bp = blueprint_library.find('sensor.other.radar')
    radar_bp.set_attribute('horizontal_fov', '15.0')
    radar_bp.set_attribute('vertical_fov', '10.0')
    radar_bp.set_attribute('range', '200.0')
    radar_bp.set_attribute('points_per_second', '3000')
    radar_bp.set_attribute('sensor_tick', '0.05')

    camera_transform = carla.Transform(
        carla.Location(x=393.5, y=260.0, z=6.0),
        carla.Rotation(pitch=-5, yaw=270.0)
    )

    radar_transform = carla.Transform(
        carla.Location(x=396.0, y=260.0, z=6.0),
        carla.Rotation(pitch=-5.0, yaw=270.0)
    )

    camera      = world.spawn_actor(camera_bp,      camera_transform)
    camera_rfov = world.spawn_actor(camera_rfov_bp, radar_transform)
    radar       = world.spawn_actor(radar_bp,        radar_transform)

    radar_actor = radar

    radar.listen(lambda data: radar_callback(data))
    camera.listen(lambda data: camera_callback(data))
    camera_rfov.listen(lambda data: camera_rfov_callback(data))

    return world, blueprint_library, camera, camera_rfov, radar


SCENARIO_MAP = {
    '1': ('scenario1', scenario_speeding,        "Nadmierna prędkość (2 pojazdy)"),
    '2': ('scenario2', scenario_single_speeding, "Nadmierna prędkość (1 pojazd)"),
    '3': ('scenario3', scenario_lane_change,     "Zmiana pasa (Cut-in)"),
    '4': ('scenario4', scenario_overtaking,      "Wyprzedzanie"),
    '5': ('scenario5_empty', None,               "Pusta droga (60s)"),
}


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    camera      = None
    camera_rfov = None
    radar       = None

    try:
        world, blueprint_library, camera, camera_rfov, radar = setup_environment(client)

        while True:
            print("\n" + "=" * 40)
            print("   ZARZĄDCA SCENARIUSZY CARLA")
            print("=" * 40)
            for key, (_, _, label) in SCENARIO_MAP.items():
                print(f"  {key}. {label}")
            print(f"  W. Zmiana pogody  [{current_weather_name}]")
            print("  0. Wyjście")

            choice = input("\nWybierz: ").strip()

            if choice == '0':
                print("Zamykanie programu...")
                break

            if choice.upper() == 'W':
                change_weather(world)
                continue

            if choice not in SCENARIO_MAP:
                print("Nieprawidłowy wybór.")
                continue

            scenario_prefix, scenario_module, label = SCENARIO_MAP[choice]
            print(f"\nUruchamianie: {label}")

            start_recording(scenario_prefix)
            try:
                if choice == '5':
                    run_empty_road(world)
                else:
                    scenario_module.run(world, blueprint_library)
            except KeyboardInterrupt:
                print("\nPrzerwano scenariusz.")
            except Exception as e:
                print(f"Błąd podczas scenariusza: {e}")
            finally:
                stop_recording()

    except Exception as e:
        print(f"\nBłąd środowiska: {e}")
    finally:
        stop_recording()

        try:
            if camera is not None:
                camera.stop()
                camera.destroy()
            if camera_rfov is not None:
                camera_rfov.stop()
                camera_rfov.destroy()
            if radar is not None:
                radar.stop()
                radar.destroy()
        except Exception:
            pass


if __name__ == '__main__':
    main()