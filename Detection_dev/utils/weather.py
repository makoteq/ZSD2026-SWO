"""
Weather utilities.

Provides weather retrieval: weather condition
classification, and road-condition-dependent stopping distance
calculations.

Attribution:
Weather data by Open-Meteo.com
https://open-meteo.com/

Data license:
CC BY 4.0
https://creativecommons.org/licenses/by/4.0/
"""
import requests
from datetime import date
from dataclasses import dataclass

# Multipliers used to adjust stopping distance calculations
# according to estimated road surface conditions.
ROAD_CONDITION_MULTIPLIER={
    "dry":    1,
    "rain":   1.7,
    "snow":   3,
    "ice":    4,
}

@dataclass
class WeatherResult:
    """
    Container for weather data and derived road condition parameters.
    """
    multiplier: float
    condition: str
    weatherCode: int
    temperature: float
    description: str

def getWeather(lat: float, lon: float, targetDate: date, time: int) -> WeatherResult:
    """
    Retrieve weather information for a given location and hour.

    Parameters:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        targetDate (date): Target date.
        time (int): Hour of the day (0-23).

    Returns:
        WeatherResult: Weather data together with a road condition multiplier.
    """

    today = date.today()

    if targetDate < today:
        url = "https://archive-api.open-meteo.com/v1/archive"
        dateStr = targetDate.strftime("%Y-%m-%d")
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        dateStr = today.strftime("%Y-%m-%d")

    params ={
        "latitude": lat,
        "longitude": lon,
        "start_date": dateStr,
        "end_date": dateStr,
        "hourly": "weathercode,temperature_2m",
        "timezone": "auto",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        weatherCodes = data["hourly"]["weathercode"]
        temperatures = data["hourly"]["temperature_2m"]

        weatherCode = weatherCodes[time]
        temperature = temperatures[time]

        condition = code_to_weather(weatherCode, temperature)
        description = f"WMO code={weatherCode}, temp={temperature:.1f}°C"

        return WeatherResult(
            multiplier=ROAD_CONDITION_MULTIPLIER[condition],
            condition=condition,
            weatherCode=weatherCode,
            temperature=temperature,
            description=description,
        )

    except Exception:
        print("error getting weather, fallback to DRY")
        return WeatherResult(
            multiplier=ROAD_CONDITION_MULTIPLIER["dry"],
            condition="dry",
            weatherCode=-1,
            temperature=0.0,
            description="fallback: no data",
        )

def calcStoppingDistance(speed: float, weatherMarkiplier: float = 1.0) -> float:
    """
    Estimate stopping distance based on vehicle speed and
    road-condition multiplier.

    Parameters:
        speed (float): Vehicle speed in m/s.
        weatherMarkiplier (float) : Multiplier representing road conditions.

    Returns:
        float: Estimated stopping distance in meters.
    """

    if weatherMarkiplier == 1:
        reaction_time = 1.5
    elif weatherMarkiplier == 1.7 or weatherMarkiplier == 3:
        reaction_time = 2.0
    else:
        reaction_time = 1.0

    if speed <= 0:
        speed = 50.0/3.6

    breaking_distance = float(reaction_time * speed) +((speed/10)*3 + (speed/10)**2)*weatherMarkiplier

    return float(breaking_distance)


def code_to_weather(code: int, temp: float) -> str:
    """
    Convert a WMO weather code and temperature into a road condition
    category: dry, rain, snow, or ice.
    """

    # Snow-related weather codes
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    # Freezing rain / ice-related weather codes
    elif code in (56, 57, 66, 67):
        return "ice"
    # Rain/freezing rain (when temperature is near 0 degrees Celsius)
    elif code in (51, 53, 55, 61, 63, 65, 80, 81, 82):
        if temp <= 2.0:
            return "ice"
        return "rain"
    else:
        return "dry"
