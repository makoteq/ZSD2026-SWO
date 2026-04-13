
from street_module import Street

RELATIVE_CSV_PATH = "../../data\\batch2\\batch2\\scenario1_speeding_2cars\\run_004\\radar_points_world.csv"

SENSOR_PITCH_DEG = 0.0
SENSOR_YAW_DEG = 0.0
SENSOR_ROLL_DEG = 0.0
CAMERA_HEIGHT_OFFSET = 6.0
MASK_Z_MIN = 0.15
MASK_Z_MAX = 5.0
MASK_Y_MIN = 70.0
MASK_Y_MAX = 120.0
INTERPOLATION_FACTOR = 2

if __name__ == "__main__":
    street = Street(RELATIVE_CSV_PATH)
    street.adjustPoints(SENSOR_PITCH_DEG, SENSOR_YAW_DEG, SENSOR_ROLL_DEG, CAMERA_HEIGHT_OFFSET)
    street.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    street.interpolateCloud(INTERPOLATION_FACTOR)
    street.findRoad()
    street.findContours()
