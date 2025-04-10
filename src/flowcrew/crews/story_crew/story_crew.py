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
class StoryCrew:
    """Story Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def story_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["story_writer"],
            llm=llm # Pass the instantiated LLM
        )

    @task
    def write_story(self) -> Task:
        return Task(
            config=self.tasks_config["write_story"],
            agent=self.story_writer() # Explicitly assign agent
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Story Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
