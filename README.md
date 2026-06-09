# Early Warning System at Pedestrian Crossings

## Project Overview
The objective of this project is to develop an early warning system at pedestrian crossings that alerts pedastrians to dangerous road situations. The system triggers an alarm 5 to 15 seconds before a vehicle reaches the crossing. Detected hazardous events include speeding, overtaking, and lane departures.

## Project Structure
Description of repository directories:

```
ZSD2026-SWO/
├── CARLA_dev/          Files for data generation in the CARLA simulation environment
├── data/               Datasets (alarm/noalarm), JSON configuration, and network weight models
├── Detection_dev/      Main detection module, algorithms, helper tools, and launch scripts
├── docs/               Complete technical documentation of the project
├── hardware/           Benchmarks, metrics, and hardware platform configuration files
├── Scenarios/          Archival versions of traffic scenario implementations
├── sensors/            Archival versions of sensor configurations
├── README.md           General information about the project in English
└── README_PL.md        General information about the project in Polish
```

## Licenses and Copyright
This project integrates various open-source components. All utilized tools and datasets are applied in accordance with their respective licensing terms and conditions.

### Utilized Licenses:
| **Tool**                  | **License**                                                              |
|-|-|
| **Python (>3.8.6)** | PSF License Version 2                                                    |
| **OpenCV (>4.5.0)** | Apache License 2.0                                                       |
| **CARLA (0.9.16)** | MIT License                                                              |
| **Ultralytics YOLO (YOLOv8)** | GNU Affero General Public License v3.0                                   |
| **Depth-Anything-V2** | Apache License 2.0                                                       |
| **Open-Meteo API** | Creative Commons Attribution 4.0 International (CC BY 4.0)               |
| **Google Coral Edge TPU** | Apache License 2.0                                                       |
| **Raspberry Pi 4 Model B** | Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) |
---
