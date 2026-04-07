"""Dataset generator for CNN training: car type classification + braking distance estimation.

This script:
- spawns all 4‑wheel vehicles available in a simulator's map
- records RGB camera crops of the vehicle’s bounding box from 4 cameras with random offset and random pitch
- stores images in train/val/test with JSON metadata including vehicle type and physics parameters.

Parts of the code are based on CARLA Simulator’s Python API examples (MIT license).
See: https://github.com/carla-simulator/carla
"""

import carla
import random
import numpy as np
import os
import queue
import cv2
import json
import time

OUTPUT_DIR = "cnn_dataset"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# Cropped image size for CNN input
IMG_SIZE = 224

# Camera sensor resolution and field of view
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
FOV = 20.0
SENSOR_TICK = 0.05 # 20 Hz camera update rate

# Capture 40 frames per vehicle instance per weather/camera combo
FRAMES_PER_VEHICLE = 40
# Maximum number of simulation ticks to try before giving up on a vehicle
MAX_ATTEMPTS = 200

# Minimum bounding box size to avoid tiny / noisy crops
MIN_BBOX_WIDTH = 40
MIN_BBOX_HEIGHT = 20

# Ensure all output directories exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# A map of predefined CARLA weather presets
WEATHER_PRESETS = {
    'ClearNoon': carla.WeatherParameters.ClearNoon,
    'ClearSunset': carla.WeatherParameters.ClearSunset,
    'CloudyNoon': carla.WeatherParameters.CloudyNoon,
    'CloudySunset': carla.WeatherParameters.CloudySunset,
    'WetNoon': carla.WeatherParameters.WetNoon,
    'WetSunset': carla.WeatherParameters.WetSunset,
    'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
    'MidRainSunset': carla.WeatherParameters.MidRainSunset,
    'HardRainNoon': carla.WeatherParameters.HardRainNoon,
    'HardRainSunset': carla.WeatherParameters.HardRainSunset,
    'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
    'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
    #'ClearNight': carla.WeatherParameters.ClearNight,
    #'CloudyNight': carla.WeatherParameters.CloudyNight,
    #'WetNight': carla.WeatherParameters.WetNight,
    #'SoftRainNight': carla.WeatherParameters.SoftRainNight,
    #'MidRainyNight': carla.WeatherParameters.MidRainyNight,
    #'HardRainNight': carla.WeatherParameters.HardRainNight,
}


def get_split_dir():
    """Split data to train/test/val - randomly assign an image with 70%/15%/15% proportions
    """
    r = random.random()
    if r < 0.7:
        return TRAIN_DIR
    elif r < 0.85:
        return VAL_DIR
    else:
        return TEST_DIR

# Calculate the camera projection matrix
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

# Calculate 2D projection of 3D coordinate
def get_image_point(loc, K, world_2_camera):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(world_2_camera, point)

    point_camera = np.array([
        point_camera[1],
        -point_camera[2],
        point_camera[0]
    ])

    if point_camera[2] <= 0:
        return None

    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[:2]

def get_bbox_crop(img, points):
    """Extract a tight bounding box crop from the RGB image.
        Args:
            img: (H, W, 3) RGB image.
            points: list of 2D points defining the projected bounding box vertices.
        Returns:
            Resized square crop (IMG_SIZE x IMG_SIZE) or None if bbox is too small.
        """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    xmin = max(0, int(min(xs)))
    ymin = max(0, int(min(ys)))
    xmax = min(IMAGE_WIDTH, int(max(xs)))
    ymax = min(IMAGE_HEIGHT, int(max(ys)))

    if xmax - xmin < MIN_BBOX_WIDTH or ymax - ymin < MIN_BBOX_HEIGHT:
        return None

    crop = img[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

def remove_environment(world):
    """Disable selected environment objects from world to reduce clutter in the dataset."""
    labels = [
        carla.CityObjectLabel.Vegetation,
        carla.CityObjectLabel.Fences,
        carla.CityObjectLabel.Poles,
        carla.CityObjectLabel.Walls,
        carla.CityObjectLabel.TrafficLight,
        carla.CityObjectLabel.TrafficSigns
    ]
    for label in labels:
        objs = world.get_environment_objects(label)
        ids = [obj.id for obj in objs]
        if ids:
            world.enable_environment_objects(ids, False)

def spawn_cameras(world, blueprint_library):
    """Spawn 5 RGB camera sensors with predefined positions to ensure dataset diversity.

    - one very close (40m)
    - one very far (200m)
    - two 100m with lateral offsets (+-4m)
    - one base position
    """

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('fov', '20.0')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('sensor_tick', '0.05')
    camera_bp.set_attribute('bloom_intensity', '0.0')
    camera_bp.set_attribute('exposure_mode', 'manual')
    camera_bp.set_attribute('exposure_compensation', '0.0')
    camera_bp.enable_postprocess_effects = False

    base_x = 393.5
    base_y = 270.0
    base_z = 6.0

    transforms = []

    # Base camera (unchanged)
    transforms.append(carla.Transform(
        carla.Location(x=base_x, y=base_y, z=base_z),
        carla.Rotation(
            pitch=-5.0 + random.uniform(-3.0, 3.0),
            yaw=270.0 + random.uniform(-2.0, 2.0)
        )
    ))

    # Close (40m)
    transforms.append(carla.Transform(
        carla.Location(x=base_x, y=base_y - 40.0, z=base_z),
        carla.Rotation(
            pitch=-5.0 + random.uniform(-3.0, 3.0),
            yaw=270.0 + random.uniform(-2.0, 2.0)
        )
    ))

    # Far (200m)
    transforms.append(carla.Transform(
        carla.Location(x=base_x, y=base_y - 200.0, z=base_z),
        carla.Rotation(
            pitch=-5.0 + random.uniform(-3.0, 3.0),
            yaw=270.0 + random.uniform(-2.0, 2.0)
        )
    ))

    # 100m left offset
    transforms.append(carla.Transform(
        carla.Location(x=base_x - 4.0, y=base_y - 100.0, z=base_z),
        carla.Rotation(
            pitch=-5.0 + random.uniform(-3.0, 3.0),
            yaw=270.0 + random.uniform(-2.0, 2.0)
        )
    ))

    # 100m right offset
    transforms.append(carla.Transform(
        carla.Location(x=base_x + 4.0, y=base_y - 100.0, z=base_z),
        carla.Rotation(
            pitch=-5.0 + random.uniform(-3.0, 3.0),
            yaw=270.0 + random.uniform(-2.0, 2.0)
        )
    ))

    cameras = []
    queues = []

    for t in transforms:
        cam = world.spawn_actor(camera_bp, t)
        q = queue.Queue()
        cam.listen(q.put)
        cameras.append(cam)
        queues.append(q)

    return cameras, queues

def main():
    """Handle CARLA connection, spawn vehicles from available blueprints, capture images with metadata.

    For each weather preset & each 4-wheel vehicle blueprint:
    -spawn the vehicle on a fixed point
    -capture valid frames
    -store data in the dataset
    """

    # Handle CARLA connection
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)

    # Setup world
    world = client.load_world("Town01")
    blueprint_library = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = SENSOR_TICK
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        carla.Location(x=400.0, y=200.0, z=6.0),
        carla.Rotation(pitch=-10.0, yaw=-105.0)
    ))

    remove_environment(world)

    # Build intrinsic camera matrix once; it is the same for all cameras
    K = build_projection_matrix(IMAGE_WIDTH, IMAGE_HEIGHT, FOV)

    # Get all 4‑wheel vehicle blueprints (cars, trucks, SUVs, etc.)
    vehicle_blueprints = [
        bp for bp in blueprint_library.filter('vehicle.*')
        if int(bp.get_attribute('number_of_wheels')) == 4
    ]

    random.shuffle(vehicle_blueprints)

    total_saved = 0

    try:
        # Loop over all weather presets
        for weather_name, weather in WEATHER_PRESETS.items():

            world.set_weather(weather)

            # Loop over all vehicle blueprints
            for bp in vehicle_blueprints:

                # Fixed spawn point for vehicles - on a road lane
                transform = carla.Transform(
                    carla.Location(x=392.0, y=65.0, z=0.4),
                    carla.Rotation(yaw=90.0)
                )

                # Try to spawn the vehicle; continue if spawn fails
                vehicle = world.try_spawn_actor(bp, transform)
                if vehicle is None:
                    continue

                vehicle.set_autopilot(True)
                physics = vehicle.get_physics_control()

                # Spawn synchronized RGB cameras
                cameras, queues = spawn_cameras(world, blueprint_library)

                time.sleep(1.0)

                frames_collected = 0
                attempts = 0

                # Turn on vehicle lights (position + low beam) if supported
                has_lights = False

                try:
                    vehicle.set_light_state(
                        carla.VehicleLightState(
                            carla.VehicleLightState.Position |
                            carla.VehicleLightState.LowBeam
                        )
                    )
                    state = vehicle.get_light_state()
                    if state != carla.VehicleLightState.NONE:
                        has_lights = True

                except:
                    has_lights = False

                # Skip vehicles without lights in difficult night/weather conditions
                restricted_conditions = [
                    'CloudyNight',
                    'MidRainyNight',
                    'SoftRainNight',
                    'HardRainNoon',
                    'HardRainSunset',
                ]

                if (weather_name in restricted_conditions) and (not has_lights):
                    for cam in cameras:
                        cam.stop()
                        cam.destroy()
                    vehicle.destroy()
                    continue

                # Collect frames until we reach FRAMES_PER_VEHICLE threshold or a timeout
                while frames_collected < FRAMES_PER_VEHICLE and attempts < MAX_ATTEMPTS:

                    world.tick()

                    for cam_id, (cam, q) in enumerate(zip(cameras, queues)):

                        try:
                            image = q.get_nowait()
                        except queue.Empty:
                            continue

                        # Convert raw RGBA image data to a numpy array
                        img = np.reshape(
                            np.copy(image.raw_data),
                            (image.height, image.width, 4)
                        )[:, :, :3]

                        # Compute world‑to‑camera transform matrix
                        world_2_camera = np.array(
                            cam.get_transform().get_inverse_matrix()
                        )

                        # Get the 3D bounding box of the vehicle in world coordinates
                        verts = vehicle.bounding_box.get_world_vertices(
                            vehicle.get_transform()
                        )

                        # Project each 3D vertex to 2D
                        points = []
                        for v in verts:
                            p = get_image_point(v, K, world_2_camera)
                            if p is not None:
                                if 0 <= p[0] < IMAGE_WIDTH and 0 <= p[1] < IMAGE_HEIGHT:
                                    points.append(p)

                        if len(points) < 4:
                            continue

                        crop = get_bbox_crop(img, points)
                        if crop is None:
                            continue

                        split_dir = get_split_dir()
                        label = bp.id.replace('.', '_')

                        label_dir = os.path.join(split_dir, label)
                        os.makedirs(label_dir, exist_ok=True)

                        # Unique filename per weather, frame index, and camera
                        filename = f"{weather_name}_{frames_collected}_cam{cam_id}.jpg"
                        path = os.path.join(label_dir, filename)

                        cv2.imwrite(path, crop)

                        # Save corresponding JSON metadata
                        meta_path = path.replace(".jpg", ".json")
                        with open(meta_path, "w") as f:
                            vehicle_type = {
                                "base_type": bp.get_attribute("base_type").as_str(),
                                "generation": bp.get_attribute("generation").as_int(),
                                "number_of_wheels": bp.get_attribute("number_of_wheels").as_int(),
                            }

                            # Extract physics parameters relevant to braking &  vehicle dynamics
                            physics = vehicle.get_physics_control()
                            physics_dict = {
                                "mass": physics.mass,
                                "drag_coefficient": physics.drag_coefficient,
                                "clutch_strength": physics.clutch_strength,
                                "gear_switch_time": physics.gear_switch_time,
                            }

                            # Save metadata
                            json.dump({
                                "type_id": bp.id,
                                "weather": weather_name,
                                "camera_id": cam_id,
                                "vehicle_type": vehicle_type,
                                "physics": physics_dict,
                                "distance_approx": abs(
                                    cam.get_transform().location.y - vehicle.get_transform().location.y)
                            }, f, indent=2)

                        frames_collected += 1
                        total_saved += 1

                        if frames_collected >= FRAMES_PER_VEHICLE:
                            break

                    attempts += 1

                # Clean up
                for cam in cameras:
                    cam.stop()
                    cam.destroy()

                vehicle.destroy()

    except KeyboardInterrupt:
        # Handle user interrupt exit
        pass

    print(f"\nImages saved: {total_saved}")

if __name__ == "__main__":
    main()