from google.adk.agents.llm_agent import Agent
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

root_agent = Agent(
    model='gemini-2.5-flash-lite',
    name='root_agent',
    description='A helpful assistant for user questions.',
    tools=[google_search],
    instruction='Answer user questions to the best of your knowledge, do a web search if you are unsure.',
)
