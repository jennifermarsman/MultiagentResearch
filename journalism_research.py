import datetime
import json
import os
from dotenv import load_dotenv
from typing import List
from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown

# Azure AI Project imports
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MessageRole, BingGroundingTool
from azure.identity import DefaultAzureCredential


# Define function to create a journalism research agent
def create_journalism_research_agent():
    # Initialize the AI Project client
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=os.environ["PROJECT_CONNECTION_STRING"],
    )

    # Get Bing connection
    bing_connection = project_client.connections.get(connection_name=os.environ["BING_CONNECTION_NAME"])
    conn_id = bing_connection.id
    print(f"Using Bing connection ID: {conn_id}")

    # Initialize Bing grounding tool
    bing = BingGroundingTool(connection_id=conn_id)

    # Create agents with specific roles
    # Create writer agent
    writer_agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="writer_assistant",
        instructions="You are a high-quality journalist agent who excels at writing a first draft of an article as well as revising the article based on feedback from the other agents. Do not just write bullet points on how you would write the article, but actually write it. You can also ask for research to be conducted on certain topics.",
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"},
    )
    print(f"Created writer agent, ID: {writer_agent.id}")

    # Create editor agent
    editor_agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="editor_agent",
        instructions="You are an expert editor. You carefully read an article and make suggestions for improvements and suggest additional topics that should be researched to improve the article quality.",
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"},
    )
    print(f"Created editor agent, ID: {editor_agent.id}")

    # Create verifier agent
    verifier_agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="verifier_agent",
        instructions="You are responsible for ensuring the article's accuracy. You should use the Bing tool to search the internet to verify any relevant facts, and explicitly approve or reject the article based on accuracy, giving your reasoning. You can ask for rewrites if you find inaccuracies.",
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"},
    )
    print(f"Created verifier agent, ID: {verifier_agent.id}")

    # Create orchestrator agent
    orchestrator_agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="orchestrator_agent",
        instructions="You are leading a journalism team that conducts research to craft high-quality articles. You ensure that the output contains an actual well-written article, not just bullet points on what or how to write the article. If the article isn't to that level yet, ask the writer for a rewrite. If the team has written a strong article with a clear point that meets the requirements, and has been reviewed by the editor, and has been fact-checked and approved by the verifier agent, then respond with a completion message.",
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"},
    )
    print(f"Created orchestrator agent, ID: {orchestrator_agent.id}")

    return project_client, writer_agent, editor_agent, verifier_agent, orchestrator_agent


def main():
    # Create agents
    project_client, writer_agent, editor_agent, verifier_agent, orchestrator_agent = create_journalism_research_agent()

    # Create a thread for communication
    with project_client:
        thread = project_client.agents.create_thread()
        print(f"Created thread, ID: {thread.id}")

        # Ask user for article topic
        console = Console()
        console.print("Welcome to the journalism research assistant.", style="bold green")
        console.print("Describe the article you want to write. You can optionally add some starting points.", style="italic")
        user_input = input("> ")

        # Add today's date to the user input
        today_date = datetime.datetime.now().strftime("%d %B %Y")
        full_prompt = f"Article topic: {user_input}\nDate: {today_date}"

        # Create initial message from user
        message = project_client.agents.create_message(
            thread_id=thread.id,
            role=MessageRole.USER,
            content=full_prompt,
        )
        print(f"Created message, ID: {message.id}")

        # Process journalism workflow with the agents
        console.print("\n--- Start research and writing process ---\n", style="bold blue")
        
        # Step 1: Writer creates initial draft
        console.print("Step 1: Writer agent creates a first draft...", style="bold magenta")
        run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=writer_agent.id)
        print_agent_response(project_client, thread.id, console, "Writer")
        
        # Step 2: Editor reviews and provides feedback
        console.print("\nStep 2: Editor agent reviews the draft...", style="bold magenta")
        run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=editor_agent.id)
        print_agent_response(project_client, thread.id, console, "Editor")
        
        # Step 3: Writer improves based on feedback
        console.print("\nStep 3: Writer agent improves the article...", style="bold magenta")
        run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=writer_agent.id)
        print_agent_response(project_client, thread.id, console, "Writer")
        
        # Step 4: Verifier checks facts
        console.print("\nStep 4: Verifier agent checks the facts...", style="bold magenta")
        run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=verifier_agent.id)
        print_agent_response(project_client, thread.id, console, "Verifier")
        
        # Step 5: Writer makes final adjustments
        console.print("\nStep 5: Writer agent makes the final adjustments...", style="bold magenta")
        run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=writer_agent.id)
        print_agent_response(project_client, thread.id, console, "Writer")
        
        # Step 6: Orchestrator provides final evaluation
        console.print("\nStep 6: Orchestrator agent provides the final evaluation...", style="bold magenta")
        run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=orchestrator_agent.id)
        print_agent_response(project_client, thread.id, console, "Orchestrator")
        
        # Cleanup
        console.print("\n--- Cleaning up resources ---", style="bold blue")
        project_client.agents.delete_agent(writer_agent.id)
        project_client.agents.delete_agent(editor_agent.id)
        project_client.agents.delete_agent(verifier_agent.id)
        project_client.agents.delete_agent(orchestrator_agent.id)
        print("Agents removed")

# Helper function to print agent responses
def print_agent_response(project_client, thread_id, console, agent_role):
    response_message = project_client.agents.list_messages(thread_id=thread_id).get_last_message_by_role(
        MessageRole.AGENT
    )
    if response_message:
        console.print(f"\n[bold magenta]{agent_role}[/bold magenta]: ")
        for text_message in response_message.text_messages:
            console.print(Markdown(text_message.text.value))
        for annotation in response_message.url_citation_annotations:
            console.print(f"[italic]Source: [{annotation.url_citation.title}]({annotation.url_citation.url})[/italic]")


# Load env variables
load_dotenv()

# Azure AI Project variables
os.environ["PROJECT_CONNECTION_STRING"] = os.getenv("PROJECT_CONNECTION_STRING", "")
os.environ["MODEL_DEPLOYMENT_NAME"] = os.getenv("AZURE_MODEL_DEPLOYMENT", "")
os.environ["BING_CONNECTION_NAME"] = os.getenv("BING_CONNECTION_NAME", "")

# Run
main()
