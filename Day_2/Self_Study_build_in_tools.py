import os
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai.types import (
    Tool,
    GenerateContentConfig,
    HttpOptions,
    UrlContext
)

# Load .env from project root before client init
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY/API_KEY. Set it in .env or environment before running.")

# Ensure Vertex AI is disabled unless explicitly enabled
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

# Create client with api_key so credentials are recognized
client = genai.Client(api_key=GOOGLE_API_KEY, http_options=HttpOptions(api_version="v1"))

model_id = "gemini-2.5-flash"
url_context_tool = Tool(
    url_context = UrlContext
)
url1 ="https://www.foodnetwork.com/recipes/ina-garten/perfect-roast-chicken-recipe-1940592"
url2 = "https://www.allrecipes.com/recipe/70679/simple-whole-roasted-chicken/"
response = client.models.generate_content(
    model=model_id,
    contents=("Compare the ingredients and cooking times from "
              f"the recipes at {url1} and {url2}"),
    config=GenerateContentConfig(
        tools=[url_context_tool],
        response_modalities=["TEXT"],
    )
)
for each in response.candidates[0].content.parts:
    print(each.text)
# For verification, you can inspect the metadata to see which URLs the model retrieved
print(response.candidates[0].url_context_metadata)