# Batch Report
21.04.2026
- Quantity of recordings: **24**
- Correct classifications: **17**
- Accuracy: **70.83%**
- False alarm rate (FAR): **33.33%**
- True alarm rate (TAR): **72.22%**


## Review:
- Improvement of overtaking dection is needed for control cases
- Failure in 'pull over' scenarios
- Overtaking in general is inconsistent
- Speed limit detection is reliable


| # | expected_folder | alarm_detected | alarm_reasons | expected_alarm | match | video_path | csv_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | alarm | True | speed_limit_exceeded(2) | True | True | alarm\10_speeding.mp4 | alarm\10_speeding.csv |
| 2 | alarm | True | speed_limit_exceeded(2) | True | True | alarm\11_speeding.mp4 | alarm\11_speeding.csv |
| 3 | alarm | True | speed_limit_exceeded(2) | True | True | alarm\12_speeding.mp4 | alarm\12_speeding.csv |
| 4 | alarm | True | speed_limit_exceeded(2) | True | True | alarm\13_overtaking.mp4 | alarm\13_overtaking.csv |
| 5 | alarm | True | speed_limit_exceeded(1) | True | True | alarm\14_overtaking.mp4 | alarm\14_overtaking.csv |
| 6 | alarm | True | overtaking_detected(6); speed_limit_exceeded(1) | True | True | alarm\15_overtaking.mp4 | alarm\15_overtaking.csv |
| 7 | alarm | True | overtaking_detected(2); speed_limit_exceeded(1) | True | True | alarm\16_overtaking.mp4 | alarm\16_overtaking.csv |
| 8 | alarm | True | overtaking_detected(1); speed_limit_exceeded(2) | True | True | alarm\17_overtaking.mp4 | alarm\17_overtaking.csv |
| 9 | alarm | True | lane_departure(1); overtaking_detected(1); speed_limit_exceeded(3) | True | True | alarm\18_overtaking.mp4 | alarm\18_overtaking.csv |
| 10 | alarm | False | none | True | False | alarm\19_pull_over.mp4 | alarm\19_pull_over.csv |
| 11 | alarm | False | none | True | False | alarm\20_pull_over.mp4 | alarm\20_pull_over.csv |
| 12 | alarm | False | none | True | False | alarm\21_pull_over.mp4 | alarm\21_pull_over.csv |
| 13 | alarm | False | none | True | False | alarm\22_pull_over.mp4 | alarm\22_pull_over.csv |
| 14 | alarm | True | lane_departure(1) | True | True | alarm\23_pull_over.mp4 | alarm\23_pull_over.csv |
| 15 | alarm | False | none | True | False | alarm\24_pull_over.mp4 | alarm\24_pull_over.csv |
| 16 | alarm | True | speed_limit_exceeded(2) | True | True | alarm\7_speeding.mp4 | alarm\7_speeding.csv |
| 17 | alarm | True | speed_limit_exceeded(2) | True | True | alarm\8_speeding.mp4 | alarm\8_speeding.csv |
| 18 | alarm | True | speed_limit_exceeded(1) | True | True | alarm\9_speeding.mp4 | alarm\9_speeding.csv |
| 19 | noalarm | False | none | False | True | noalarm\1_control.mp4 | noalarm\1_control.csv |
| 20 | noalarm | False | none | False | True | noalarm\2_control.mp4 | noalarm\2_control.csv |
| 21 | noalarm | False | none | False | True | noalarm\3_control.mp4 | noalarm\3_control.csv |
| 22 | noalarm | False | none | False | True | noalarm\4_control.mp4 | noalarm\4_control.csv |
| 23 | noalarm | True | lane_departure(10) | False | False | noalarm\5_control.mp4 | noalarm\5_control.csv |
| 24 | noalarm | True | lane_departure(3) | False | False | noalarm\6_control.mp4 | noalarm\6_control.csv |

