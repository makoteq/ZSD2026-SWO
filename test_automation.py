import carla
import time

def spawn_traffic(world, blueprint_library):
    vehicles = []
    # Vehicle 1
    v1_model = blueprint_library.filter('model3')[0]
    spawn_p1 = carla.Transform(carla.Location(x=20.0, y=191.5, z=0.5), carla.Rotation(yaw=0.0))
    v1 = world.spawn_actor(v1_model, spawn_p1)
    vehicles.append(v1)

    # Vehicle 2
    v2_model = blueprint_library.filter('tt')[0]
    spawn_p2 = carla.Transform(carla.Location(x=10.0, y=188.0, z=0.5), carla.Rotation(yaw=0.0))
    v2 = world.spawn_actor(v2_model, spawn_p2)
    vehicles.append(v2)

    return vehicles, v1, v2

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    try:
        print("loading map and cleaning...")
        world = client.load_world('Town02')
        world.set_weather(carla.WeatherParameters.ClearNoon)

        # cleaning map
        labels_to_hide = [
            carla.CityObjectLabel.Buildings,
            carla.CityObjectLabel.Fences,
            carla.CityObjectLabel.Vegetation,
            carla.CityObjectLabel.Other,
            carla.CityObjectLabel.Walls
        ]
        all_to_hide = [obj.id for label in labels_to_hide for obj in world.get_environment_objects(label)]
        world.enable_environment_objects(all_to_hide, False)


        #Spectator
        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            carla.Location(x=182.5, y=189.5, z=5.0),
            carla.Rotation(pitch=-10.0, yaw=180.0)
        ))
        blueprint_library = world.get_blueprint_library()
        vehicles, v1, v2 = spawn_traffic(world, blueprint_library)
        target_speed = 50/3.6
        crossing = 177.0
        v2_velo_boost = 0
        while True:
            for v in vehicles:
                if v == vehicles[1]:
                    v.set_target_velocity(carla.Vector3D(x=target_speed+v2_velo_boost, y=0.0, z=0.0))
                else:
                    v.set_target_velocity(carla.Vector3D(x=target_speed, y=0.0, z=0.0))

            loc1 = v1.get_location()
            loc2 = v2.get_location()
            velocity1 = v1.get_velocity()
            velocity2 = v2.get_velocity()
            speed1 = 3.6 * (velocity1.x ** 2 + velocity1.y ** 2 + velocity1.z ** 2) ** 0.5
            speed2 = 3.6 * (velocity2.x ** 2 + velocity2.y ** 2 + velocity2.z ** 2) ** 0.5

            print(f"V1: {speed1:5.1f} km/h| "
                  f"V2: {speed2:5.1f} km/h", end="\r")

            if loc1.x > crossing and loc2.x > crossing:
                v2_velo_boost += 1/3.6
                for v in vehicles:
                    if v.is_alive:
                        v.destroy()

                vehicles, v1, v2 = spawn_traffic(world, blueprint_library)
                time.sleep(0.1)
            time.sleep(0.05)


    except KeyboardInterrupt:
        print("\nStopping simulation...")
    except Exception as e:
        print(f"\nerror: {e}")
    finally:
        for v in vehicles:
            if v is not None:
                v.destroy()
            print("finished")


if __name__ == '__main__':
    main()