# System wczesnego ostrzegania na przejściu dla pieszych

## Opis Projektu
Celem projektu jest opracowanie systemu wczesnego ostrzegania na przejściu dla pieszych, który informuje o niebezpiecznych sytuacjach drogowych podnosząc alarm na 5 do 15 sekund przed dotarciem pojazdu do przejścia. Wykrywane zdarzenia niebezpieczne obejmują przekroczenie prędkośći, wyprzedzanie lub zjeżdżanie z toru ruchu

## Struktura Projektu
Opis katalogów w repozytorium:
```
ZSD2026-SWO/
├── CARLA_dev/          Pliki do generacji danych w środowisku symulacyjnym CARLA
├── data/               Zbiory danych (alarm/noalarm), konfiguracja JSON oraz modele wag sieci
├── Detection_dev/      Główny moduł detekcji, algorytmy, narzędzia pomocnicze i skrypty uruchomieniowe
├── docs/               Kompletna dokumentacja techniczna projektu
├── hardware/           Benchmarki, metryki oraz pliki konfiguracyjne platform sprzętowych
├── Scenarios/          Archiwalne wersje implementacji scenariuszy drogowych
├── sensors/            Archiwalne wersje konfiguracji sensorów
├── README.md           Ogólne informacje o projekcie w języku angielskim
└── README_PL.md        Ogólne informacje o projekcie w języku polskim
```

## Licencje i Prawa Autorskie
Realizacja projektu wykorzystuje integrację komponentów open-source, przy czym wszystkie użyte narzędzia oraz zbiory danych są stosowane zgodnie z obowiązującymi dla nich warunkami licencyjnymi.
### wykorzystywane licencje:
|**Narzędzie**|**Licencja**|
|-|-|
| **Python (>3.8.6)** | PSF License Version 2 |
| **OpenCV (>4.5.0)** | Apache License 2.0 |
| **CARLA (0.9.16)** | MIT License |
| **Ultralytics YOLO (YOLOv8)** | GNU Affero General Public License v3.0 |
| **Depth-Anything-V2** | Apache License 2.0 |
| **Open-Meteo API** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Google Coral Edge TPU** | Apache License 2.0 |
| **Raspberry Pi 4 Model B** | Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) |
---
