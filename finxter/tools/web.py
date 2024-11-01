import sys
from time import time
import asyncio
import json
from typing import List
import aiohttp
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field


def parse_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in ["nav", "footer", "aside", "script", "style", "img", "header"]:
        for match in soup.find_all(tag):
            # remove all the unnecessary content
            match.decompose()

    text_content = soup.get_text()
    text_content = " ".join(text_content.split())
    return text_content[:8_000]


async def get_webpage_content(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html_content = await response.text()

    text_content = parse_html(html_content)
    print(f"URL: {url} - fetched successfully.")
    return text_content


class ResearchInput(BaseModel):
    research_urls: List[str] = Field(description="Must be a list of valid URLs.")


@tool("research", args_schema=ResearchInput)
async def research(research_urls: List[str]) -> str:
    """Get content of provided URLs for research purposes."""
    tasks = [asyncio.create_task(get_webpage_content(url)) for url in research_urls]
    # in case one of the task fails it does not stop the others (return_exceptions=True)
    contents = await asyncio.gather(*tasks, return_exceptions=True)
    return json.dumps(contents)


if __name__ == "__main__":

    TEST_URLS = [
        "https://en.wikipedia.org/wiki/SpongeBob_SquarePants",
        "https://en.wikipedia.org/wiki/Sephen_Hillenburg",
        "https://en.wikipedia.org/wiki/The_SpongeBob_Movie:_Sponge_Out_of_Water",
    ]

    async def main():
        result = await research.ainvoke({"research_urls": TEST_URLS})
        with open("test.json", "w", encoding="UTF-8") as f:
            json.dump(result, f)

    start_time = time()
    asyncio.run(main())
    print(f"Async time: {(time() -  start_time):.2f} seconds.")
