"""
Function takes location (latitude and longitude) and time and date
and returns weather in selected place and selected time

"""
import requests
from enum import Enum
#from typing import Tuple
from datetime import date

class Weather(Enum):
    DRY = "dry"
    RAIN = "rain"
    SNOW = "snow"
    ICE = "ice"

ROAD_CONDITION_MULTIPLAYER={
    Weather.DRY:  1,
    Weather.RAIN: 1.7,
    Weather.SNOW: 3,
    Weather.ICE: 4,

}

def getWeather(lat, lon, targetDate, time):

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

        weatherCodes = data["weather"]["weathercode"]
        temperatures = data["hourly"]["temperature_2m"]

        weatherCode = weatherCodes[time]
        temperature = temperatures[time]

        condition = code_to_weather(weatherCode, temperature)
        description = f"WMO code={weatherCode}, temp={temperature:.1f}°C"

        return condition, description

    except Exception as e:
        print("error getting weather, fallback to DRY")
        return Weather.DRY, "deafult"


def code_to_weather(code, temp):
    #snow
    if code in (71, 73, 75, 77, 85, 86):
        return Weather.SNOW
    #Ice
    elif code in (56, 57, 66, 67):
        return Weather.ICE
    #rain or if below 2deg - ice
    elif code in (51, 53, 55, 61, 63, 65, 80, 81, 82):
        if temp <= 2.0:
            return Weather.ICE
        return Weather.RAIN
    else:
        return Weather.DRY
