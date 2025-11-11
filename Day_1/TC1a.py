import asyncio
import os
from venv import create
from dotenv import load_dotenv
from google import genai
import google.adk.cli


from pathlib import Path

from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "Missing GOOGLE_API_KEY (or API_KEY) in .env file. Add it before running this script."
    )


os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

print("✅ Gemini API key loaded from .env and environment configured.")
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

print("✅ ADK components imported successfully.")

# Define helper functions that will be reused throughout the notebook

try:
    from IPython.display import display, HTML
    from jupyter_server.serverapp import list_running_servers
except ImportError as IPYTHON_IMPORT_ERR:
    display = HTML = None
    list_running_servers = None
else:
    IPYTHON_IMPORT_ERR = None

# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    if IPYTHON_IMPORT_ERR is not None:
        raise RuntimeError(
            "IPython and jupyter-server are required for get_adk_proxy_url(). "
            "Install them with `pip install ipython jupyter-server`."
        ) from IPYTHON_IMPORT_ERR

    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]['base_url']

    try:
        path_parts = baseURL.split('/')
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>⚠️ IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) ↗
        </a>
    </div>
    """

    if display and HTML:
        display(HTML(styled_html))
    else:
        print("Open the ADK Web UI at:", url)

    return url_prefix

print("✅ Helper functions defined.")

root_agent = Agent(
    name="helpful_assistant",
    model="gemini-2.5-flash-lite",
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)

print("✅ Root Agent defined.")

runner = InMemoryRunner(agent=root_agent)

print("✅ Runner created.")


async def main() -> None:
    response = await runner.run_debug(
        "Where does 'Korevora' mean, is this related to any name ?"
    )
    print("\n=== Agent Response ===")
    print(response.text if hasattr(response, "text") else response)


if __name__ == "__main__":
    asyncio.run(main())
