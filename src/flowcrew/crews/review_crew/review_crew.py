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
class ReviewCrew:
    """Review Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def story_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["story_reviewer"],
            llm=llm # Pass the instantiated LLM
        )

    @task
    def review_story(self) -> Task:
        return Task(
            config=self.tasks_config["review_story"],
            agent=self.story_reviewer() # Explicitly assign agent
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Review Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
