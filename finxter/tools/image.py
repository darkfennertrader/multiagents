import uuid
from pathlib import Path
import requests
from langchain.tools import tool
from openai import OpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

from config import set_environment_variables

IMAGE_DIRECTORY = Path(__file__).parent.parent / "images"
CLIENT = OpenAI()


def image_downloader(image_url: str | None) -> str:
    if image_url is None:
        return "No image URL returned from the API."
    response = requests.get(image_url, timeout=30)
    if response.status_code != 200:
        return "Could not download image from URL."
    unique_id = uuid.uuid4()
    image_path = IMAGE_DIRECTORY / f"{unique_id}.png"
    with open(image_path, "wb") as file:
        file.write(response.content)
    return str(image_path)


class GenerateImageInput(BaseModel):
    image_description: str = Field(
        description="A detailed description of the desired image."
    )


@tool("generate_image", args_schema=GenerateImageInput)
def generate_image(image_description: str) -> str:
    """Generate an image based on a detailed description."""
    response = CLIENT.images.generate(
        model="dall-e-3",
        prompt=image_description,
        size="1024x1024",
        quality="standard",  # standard or hd
        n=1,
    )
    image_url = response.data[0].url
    return image_downloader(image_url)


if __name__ == "__main__":
    print(
        generate_image.run(
            "An image of Margareth Tatcher the former british prime minister"
        )
    )
