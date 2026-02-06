from crewai import Task, Crew
from agents.hello_agent import hello_agent

hello_agent.max_tokens = 50  # cost safety

hello_task = Task(
    description="Say hello and explain what an agent is in one sentence.",
    expected_output="A single concise sentence explaining what an agent is.",
    agent=hello_agent
)

crew = Crew(
    agents=[hello_agent],
    tasks=[hello_task],
    verbose=True
)

result = crew.kickoff()
print("\nFINAL RESULT:\n", result)
