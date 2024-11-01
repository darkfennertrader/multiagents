from json import dumps
import requests
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from config import set_environment_variables

set_environment_variables()

weather = OpenWeatherMapAPIWrapper()  # type: ignore


class WeatherInput(BaseModel):
    location: str = Field(description="Must be a valid location in city format.")


@tool("get_weather", args_schema=WeatherInput)
def get_weather(location: str) -> str:
    """Get the current weather for a specified location."""
    if not location:
        return "Please provide a location and call the get_weather function again."
    response = weather.run(location)
    return response


if __name__ == "__main__":
    print(get_weather.run("London"))
