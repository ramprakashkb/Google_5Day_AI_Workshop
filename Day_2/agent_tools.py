import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool, google_search
from google.adk.runners import InMemoryRunner

#Enable API Key from the environment file
PROJECT_ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
DAY1_SAMPLE_ENV = Path(__file__).resolve().parents[1] / "Day_1" / "sample-agent" / ".env"
for _env in (PROJECT_ROOT_ENV, DAY1_SAMPLE_ENV):
    if _env.exists():
        load_dotenv(_env, override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY/API_KEY. Set it in .env or environment before running.")

# Ensure Vertex AI is disabled unless explicitly enabled
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")


tool_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="capital_agent_a",
    description="Answer users request using available tool",
    instruction="""If the user gives you the name of a country or a state (e.g.
                Tennessee or New South Wales), answer with the name of the capital city of that
                country or state. Otherwise, User available tools to answer the other questions. But give the user the clear messge
                that you are specialized only to answer capital city of any country or state.""",
    tools=[google_search]
)

user_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="user_advice_agent",
    description="Answers user questions and gives advice",
    instruction="""Use the tools you have available to answer the user's questions""",
    tools=[AgentTool(agent=tool_agent)]
)

runner = InMemoryRunner(agent=user_agent)
print("✅ Runner created.")

async def main() -> None:
    print("\n=== Agent Request ===")
    response = await runner.run_debug("What is the temperature of Chennai?")

if __name__ == "__main__":
    asyncio.run(main())