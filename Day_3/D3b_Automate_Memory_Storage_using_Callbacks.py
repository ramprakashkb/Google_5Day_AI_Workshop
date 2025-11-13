import os
import asyncio
import uuid
import base64
import sqlite3

from pathlib import Path
from dotenv import load_dotenv
from google.adk.code_executors import BuiltInCodeExecutor
from google.genai import types
from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool
from IPython.display import display, Image as IPImage

from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.sessions import DatabaseSessionService

from typing import Any, Dict
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory



# Load .env from project root before client init
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

# Fetch API key from environment file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY/API_KEY. Set it in .env or environment before running.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

"""

When working with LLMs, you may encounter transient errors like rate limits or temporary service unavailability. 
Retry options automatically handle these failures by retrying the request with exponential backoff.

"""
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


async def run_session(
        runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model: > {text}")


print("✅ Helper functions defined.")

# Initialize memory service
memory_service = (
    InMemoryMemoryService()
)  # ADK's built-in Memory Service for development and testing

# Create Session Service
session_service = InMemorySessionService()  # Handles conversations

# Define constants used throughout the notebook
APP_NAME = "MemoryDemoApp"
USER_ID = "demo_user"

async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )


print("✅ Callback created.")

# Agent with automatic memory saving
auto_memory_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="AutoMemoryAgent",
    instruction="Answer user questions.",
    tools=[preload_memory],
    after_agent_callback=auto_save_to_memory,  # Saves after each turn!
)

print("✅ Agent created with automatic memory saving!")



# Create a runner for the auto-save agent
# This connects our automated agent to the session and memory services
auto_runner = Runner(
    agent=auto_memory_agent,  # Use the agent with callback + preload_memory
    app_name=APP_NAME,
    session_service=session_service,  # Same services from Section 3
    memory_service=memory_service,
)

print("✅ Runner created.")

async def main() -> None:

    # Test 1: Tell the agent about a gift (first conversation)
    # The callback will automatically save this to memory when the turn completes
    await run_session(
        auto_runner,
        "I gifted a new toy to my nephew on his 1st birthday!",
        "auto-save-test",
    )

    # Test 2: Ask about the gift in a NEW session (second conversation)
    # The agent should retrieve the memory using preload_memory and answer correctly
    await run_session(
        auto_runner,
        "What did I gift my nephew?",
        "auto-save-test-2",  # Different session ID - proves memory works across sessions!
    )

if __name__ == "__main__":
    asyncio.run(main())