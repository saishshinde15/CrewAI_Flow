#!/usr/bin/env python
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from crewai.flow import Flow, listen, start

# Import the new crews
from flowcrew.crews.idea_crew.idea_crew import IdeaCrew
from flowcrew.crews.story_crew.story_crew import StoryCrew
from flowcrew.crews.review_crew.review_crew import ReviewCrew

load_dotenv() # Load environment variables like GEMINI_API_KEY

# Define the state for the new story flow
class StoryFlowState(BaseModel):
    theme: str = "a lost robot searching for home in a magical forest"
    story_idea: str = Field(default="")
    story: str = Field(default="")
    review: str = Field(default="")


# Define the new multi-crew flow
class StoryFlow(Flow[StoryFlowState]):

    @start()
    def generate_story_idea(self):
        print(f"--- Generating Story Idea for Theme: {self.state.theme} ---")
        result = (
            IdeaCrew()
            .crew()
            .kickoff(inputs={"theme": self.state.theme})
        )
        print("\n--- Story Idea Generated ---")
        print(result.raw)
        self.state.story_idea = result.raw

    @listen(generate_story_idea)
    def write_the_story(self):
        print("\n--- Writing Story based on Idea ---")
        result = (
            StoryCrew()
            .crew()
            .kickoff(inputs={"story_idea": self.state.story_idea})
        )
        print("\n--- Story Written ---")
        print(result.raw)
        self.state.story = result.raw

    @listen(write_the_story)
    def review_the_story(self):
        print("\n--- Reviewing Story ---")
        result = (
            ReviewCrew()
            .crew()
            .kickoff(inputs={"story": self.state.story})
        )
        print("\n--- Story Reviewed ---")
        print(result.raw)
        self.state.review = result.raw

    @listen(review_the_story)
    def finalize_output(self):
        print("\n--- Flow Complete ---")
        print(f"Theme: {self.state.theme}")
        print(f"\nGenerated Idea:\n{self.state.story_idea}")
        print(f"\nWritten Story:\n{self.state.story}")
        print(f"\nReview:\n{self.state.review}")

        # Optionally save the output
        output_filename = "story_output.md"
        with open(output_filename, "w") as f:
            f.write(f"# Story Generation Flow Output\n\n")
            f.write(f"## Theme\n{self.state.theme}\n\n")
            f.write(f"## Generated Idea\n{self.state.story_idea}\n\n")
            f.write(f"## Written Story\n{self.state.story}\n\n")
            f.write(f"## Review\n{self.state.review}\n")
        print(f"\nOutput saved to {output_filename}")


# Update kickoff and plot functions for the new flow
def kickoff():
    """Runs the StoryFlow."""
    story_flow = StoryFlow()
    story_flow.kickoff()


def plot():
    """Generates a plot diagram for the StoryFlow."""
    story_flow = StoryFlow()
    story_flow.plot()


if __name__ == "__main__":
    # Ensure API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in .env file.")
        print("Please add your Gemini API key to the .env file.")
    else:
        print("Starting the Story Generation Flow...")
        kickoff()
        # Uncomment the line below if you want to generate the plot diagram
        # plot()
