import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

spectator = world.get_spectator()
t = spectator.get_transform()

print(f"Location: x={t.location.x:.3f}, y={t.location.y:.3f}, z={t.location.z:.3f}")
print(f"Rotation: pitch={t.rotation.pitch:.3f}, yaw={t.rotation.yaw:.3f}, roll={t.rotation.roll:.3f}")