import carla
import time
import numpy as np
import cv2
import csv
import os
from datetime import datetime

client = carla.Client('localhost', 2000)
client.set_timeout(60.0)

vehicle = None
radar = None
RGB = None

session_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir   = f"output_{session_id}"
os.makedirs(output_dir, exist_ok=True)

radar_log_path = os.path.join(output_dir, "radar_log.csv")
video_path     = os.path.join(output_dir, "rgb_with_radar.avi")

latest_radar = {
    "max_speed_kmh": 0.0,
    "target_count":  0,
    "min_depth_m":   None,
}

video_writer = None
VIDEO_FPS    = 20.0

radar_csv_file   = open(radar_log_path, "w", newline="")
radar_csv_writer = csv.writer(radar_csv_file)
radar_csv_writer.writerow([
    "timestamp_s", "frame",
    "velocity_ms", "azimuth_deg", "altitude_deg", "depth_m",
    "approaching", "speed_kmh"
])

try:
    world = client.load_world('Town01')

    blueprints = world.get_blueprint_library()
    vehicles = blueprints.filter('vehicle.*')

    world.set_weather(carla.WeatherParameters.ClearNoon)
    spawn_points = world.get_map().get_spawn_points()

    vehicle_bp = world.get_blueprint_library().find('vehicle.ford.mustang')
    vehicle_bp.set_attribute('color', '200,0,10')

    vehicle_waypoint = world.get_map().get_waypoint(spawn_points[0].location)
    lane_dir = vehicle_waypoint.transform.get_forward_vector()
    spawn_location = spawn_points[0].location + lane_dir * (-50.0)

    vehicle_start_point = carla.Transform(spawn_location, spawn_points[0].rotation)
    start_point = spawn_points[0]

    radar_bp = world.get_blueprint_library().find('sensor.other.radar')
    radar_bp.set_attribute('horizontal_fov', '30')
    radar_bp.set_attribute('vertical_fov', '15')
    radar_bp.set_attribute('range', '200')
    radar_bp.set_attribute('points_per_second', '2000')

    RGB_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    RGB_bp.set_attribute('image_size_x', '800')
    RGB_bp.set_attribute('image_size_y', '600')
    RGB_bp.set_attribute('fov', '90')
    RGB_bp.set_attribute('sensor_tick', '0.0')

    carla_map = world.get_map()
    waypoint = carla_map.get_waypoint(
        start_point.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    lane_transform = waypoint.transform
    right_vec = lane_transform.get_right_vector()

    radar_transform = carla.Transform(
        lane_transform.location + right_vec * 3.0 + carla.Location(z=1.0),
        carla.Rotation(
            pitch=0.0,
            yaw=lane_transform.rotation.yaw + 180.0,
            roll=0.0
        )
    )

    radar = world.spawn_actor(radar_bp, radar_transform)

    rgb_transform = carla.Transform(
        lane_transform.location + right_vec + carla.Location(z=5.0),
        carla.Rotation(
            pitch=0.0,
            yaw=lane_transform.rotation.yaw + 180.0,
            roll=0.0
        )
    )

    RGB = world.spawn_actor(RGB_bp, rgb_transform)

    def radar_callback(radar_data):
        points = np.frombuffer(radar_data.raw_data, dtype=np.float32)
        points = np.reshape(points, (len(radar_data), 4))

        velocities = points[:, 0]
        azimuth = points[:, 1]
        altitude = points[:, 2]
        depth = points[:, 3]

        approaching = (velocities < -2.0) & (depth < 200)

        ts = radar_data.timestamp
        for i in range(len(velocities)):
            is_app   = bool(approaching[i])
            speed_kh = -velocities[i] * 3.6 if is_app else 0.0
            radar_csv_writer.writerow([
                f"{ts:.4f}", radar_data.frame,
                f"{velocities[i]:.3f}",
                f"{np.degrees(azimuth[i]):.2f}",
                f"{np.degrees(altitude[i]):.2f}",
                f"{depth[i]:.2f}",
                int(is_app), f"{speed_kh:.1f}"
            ])
        radar_csv_file.flush()

        if np.any(approaching):
            speeds = -velocities[approaching] * 3.6
            latest_radar["max_speed_kmh"] = float(np.max(speeds))
            latest_radar["target_count"]  = int(np.sum(approaching))
            latest_radar["min_depth_m"]   = float(np.min(depth[approaching]))
        else:
            latest_radar["max_speed_kmh"] = 0.0
            latest_radar["target_count"]  = 0
            latest_radar["min_depth_m"]   = None

    radar.listen(radar_callback)

    def rgb_callback(image):
        global video_writer

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        rgb = array[:, :, :3]

        bgr = rgb[:, :, ::-1].copy()

        if video_writer is None:
            fourcc       = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(
                video_path, fourcc, VIDEO_FPS,
                (image.width, image.height)
            )

        overlay    = bgr.copy()
        bar_height = 70
        cv2.rectangle(overlay, (0, 0), (image.width, bar_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, bgr, 0.45, 0, bgr)

        font      = cv2.FONT_HERSHEY_SIMPLEX
        max_speed = latest_radar["max_speed_kmh"]
        n_targets = latest_radar["target_count"]
        min_depth = latest_radar["min_depth_m"]
        color     = (0, 60, 255) if n_targets > 0 else (0, 220, 80)

        speed_txt  = f"Radar: {max_speed:.1f} km/h" if n_targets > 0 else "Radar: -- km/h"
        target_txt = f"Targets: {n_targets}"
        depth_txt  = f"Closest: {min_depth:.1f} m" if min_depth is not None else "Closest: -- m"

        cv2.putText(bgr, speed_txt,  (15, 28),  font, 0.85, color, 2, cv2.LINE_AA)
        cv2.putText(bgr, target_txt, (15, 56),  font, 0.70, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(bgr, depth_txt,  (220, 56), font, 0.70, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"Frame {image.frame}",
                    (image.width - 160, image.height - 15),
                    font, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

        video_writer.write(bgr)

    RGB.listen(rgb_callback)

    spectator = world.get_spectator()

    spectator_offset = carla.Transform(
        carla.Location(
            x=start_point.location.x,
            y=start_point.location.y - 10,
            z=start_point.location.z + 10
        ),
        carla.Rotation(pitch=-45, yaw=start_point.rotation.yaw, roll=0)
    )

    vehicle = world.try_spawn_actor(vehicle_bp, vehicle_start_point)
    spectator.set_transform(spectator_offset)

    if vehicle is not None:
        tm = client.get_trafficmanager(8000)
        vehicle.set_autopilot(True, tm.get_port())

        while True:
            time.sleep(0.1)

except KeyboardInterrupt:
    vehicle = None
    radar = None
finally:
    if radar is not None:
        radar.destroy()
    if RGB is not None:
        RGB.destroy()
    if vehicle is not None:
        vehicle.destroy()
        vehicle = None
        print("Actors destroyed. Map is clean.")

    radar_csv_file.close()
    if video_writer is not None:
        video_writer.release()