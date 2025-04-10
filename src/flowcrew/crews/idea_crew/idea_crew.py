import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

load_dotenv()

# Instantiate the LLM using CrewAI's LLM class and the exact model name
llm = LLM(
    model=os.environ["GEMINI_MODEL_NAME"],
    temperature=0.5,
)


@CrewBase
class IdeaCrew:
    """Idea Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def idea_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["idea_generator"],
            llm=llm # Pass the instantiated LLM
        )

    @task
    def generate_idea(self) -> Task:
        return Task(
            config=self.tasks_config["generate_idea"],
            agent=self.idea_generator() # Explicitly assign agent here
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Idea Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
