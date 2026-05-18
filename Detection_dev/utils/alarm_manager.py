import cv2

class AlarmManager:
    def __init__(self):
        self.current_level = 0
        self.reason = ""
        self.alarm_end_time = 0.0
        self.radar_disable_end_time = 0.0
        self.display_time = 3.0

    def trigger(self, level: int, reason: str, disable_radar_duration: float, current_time: float):
        if level >= self.current_level or current_time > self.alarm_end_time:
            self.current_level = level
            self.reason = reason
            self.alarm_end_time = current_time + self.display_time
            self.radar_disable_end_time = current_time + disable_radar_duration
            print(f"TIME: {current_time:.2f}s Alarm level: {level}: {reason}")
            if disable_radar_duration > 0:
                print(f" -> Radar disabled to {self.radar_disable_end_time:.2f}s")

    def is_radar_disabled(self, current_time: float) -> bool:
        return current_time < self.radar_disable_end_time

    def draw(self, frame, current_time: float):
        if current_time <= self.alarm_end_time:
            color = (0, 0, 255) if self.current_level >= 2 else (0, 165, 255)
            text = f"ALARM LVL {self.current_level}: {self.reason}"

            cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        else:
            self.current_level = 0