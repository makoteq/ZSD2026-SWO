import carla
import os
import sys

import scenario_speeding
import scenario_single_speeding
import scenario_lane_change
import scenario_overtaking
import scenario_normal_traffic

def setup_environment(client):
    print("Ładowanie mapy i ustawianie środowiska...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xodr_path = os.path.join(base_dir, '..', '..', 'HDMaps', '200m_oneway.xodr')

    if not os.path.exists(xodr_path):
        print(f"Błąd krytyczny: Nie znaleziono pliku {xodr_path}")
        sys.exit(1)

    with open(xodr_path, 'r', encoding='utf-8') as f:
        xodr_content = f.read()

    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=500.0,
        wall_height=0.0,
        additional_width=0.0,
        smooth_junctions=True,
        enable_mesh_visibility=True
    )

    world = client.generate_opendrive_world(xodr_content, params)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        carla.Location(x=195.0, y=3.5, z=10.0),
        carla.Rotation(pitch=-10.0, yaw=180.0)
    ))

    return world, world.get_blueprint_library()


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    try:
        world, blueprint_library = setup_environment(client)

        while True:
            print("\n" + "=" * 40)
            print("   ZARZĄDCA SCENARIUSZY CARLA")
            print("=" * 40)
            print("1. Scenariusz: Nadmierna prędkość (2 pojazdy)")
            print("2. Scenariusz: Nadmierna prędkość (1 pojazd)")
            print("3. Scenariusz: Zmiana pasa (Cut-in)")
            print("4. Scenariusz: Wyprzedzanie")
            print("5. Scenariusz: Normalny ruch uliczny (Nieskończony)")
            print("0. Wyjście")

            choice = input("Wybierz numer scenariusza do uruchomienia: ")

            if choice == '1':
                scenario_speeding.run(world, blueprint_library)
            elif choice == '2':
                scenario_single_speeding.run(world, blueprint_library)
            elif choice == '3':
                scenario_lane_change.run(world, blueprint_library)
            elif choice == '4':
                scenario_overtaking.run(world, blueprint_library)
            elif choice == '5':
                scenario_normal_traffic.run(world, blueprint_library)
            elif choice == '0':
                print("Zamykanie programu...")
                break
            else:
                print("Nieprawidłowy wybór. Spróbuj ponownie.")

    except Exception as e:
        print(f"\nBłąd środowiska: {e}")


if __name__ == '__main__':
    main()