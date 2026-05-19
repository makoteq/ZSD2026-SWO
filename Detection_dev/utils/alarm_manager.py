import cv2

class AlarmManager:
    def __init__(self):
        self.current_level = 0
        self.level1_reason = ""
        self.level2_reason = ""
        self.level1_end_time = 0.0
        self.level2_end_time = 0.0
        self.radar_disable_end_time = 0.0
        self.display_time = 3.0

    def trigger(self, level: int, reason: str, disable_radar_duration: float, current_time: float):
        if level == 2:
            self.level2_reason = reason
            self.level2_end_time = current_time + self.display_time
            print(f"TIME: {current_time:.2f}s Alarm level: {level}: {reason}")
        elif level == 1:
            self.level1_reason = reason
            self.level1_end_time = current_time + self.display_time
            print(f"TIME: {current_time:.2f}s Alarm level: {level}: {reason}")

        if disable_radar_duration > 0:
            self.radar_disable_end_time = max(self.radar_disable_end_time, current_time + disable_radar_duration)
            print(f" -> Radar disabled to {self.radar_disable_end_time:.2f}s")

    def is_radar_disabled(self, current_time: float) -> bool:
        return current_time < self.radar_disable_end_time

    def draw(self, frame, current_time: float):
        level2_active = current_time <= self.level2_end_time
        level1_active = current_time <= self.level1_end_time

        if level2_active:
            cv2.putText(frame, f"ALARM LVL 2: {self.level2_reason}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if level1_active:
            y_pos = 140 if level2_active else 100
            cv2.putText(frame, f"ALARM LVL 1: {self.level1_reason}", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

        if not level1_active and not level2_active:
            self.current_level = 0
        else:
            self.current_level = 2 if level2_active else 1