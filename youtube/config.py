import os
from datetime import date
from dotenv import load_dotenv


# Load the environment variables from the .env file
load_dotenv()


def set_environment_variables(project_name: str = "") -> None:

    if not project_name:
        project_name = f"Test_{date.today()}"

    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
    os.environ["LANGCHAIN_PROJECT"] = project_name
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
    ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
