from crewai import Agent

hello_agent = Agent(
    role="Hello Agent",
    goal="Verify CrewAI setup works",
    backstory="You are a minimal test agent.",
    verbose=True
)
