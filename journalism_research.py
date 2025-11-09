import asyncio
import datetime
import json
import requests
import os
from dotenv import load_dotenv
from typing import List, Sequence, Optional
from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown

# Microsoft Agent Framework imports
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential


# Tool to search the web using Bing
async def get_bing_snippet(query: str) -> str:
    #Perform a web search using the Bing Web Search API.
    # Set the parameters for the API request.
    count = 3       # Number of search results to return
    params = {
        'q': query,
        'count': count,
    }

    # Set the headers for the API request, including the subscription key.
    headers = {
        'Ocp-Apim-Subscription-Key': bing_api_key,
    }

    # Make the API request.
    response = requests.get(bing_endpoint, params=params, headers=headers)
    
    # Check if the request was successful (HTTP status code 200).
    if response.status_code == 200:
        search_results = response.json()
        # Extract and structure the search results.
        results_list = []
        for result in search_results['webPages']['value']:
            result_tuple = (result['name'], result['snippet'], result['url'])
            results_list.append(result_tuple)
        return json.dumps(results_list)
    else:
        error = f"Error: {response.status_code} - {response.text}"
        print(error)
        return error


async def main() -> None:
    # Create Azure OpenAI chat client for all agents
    # Try to use Azure CLI credentials, fall back to API key
    try:
        credential = DefaultAzureCredential()
        chat_client = AzureOpenAIChatClient(
            endpoint=azure_oai_endpoint,
            model=azure_model_deployment,
            api_version=azure_api_version,
            credential=credential,
        )
    except Exception as e:
        # Fall back to API key authentication
        chat_client = AzureOpenAIChatClient(
            endpoint=azure_oai_endpoint,
            model=azure_model_deployment,
            api_version=azure_api_version,
            api_key=azure_oai_key,
        )

    # Define agents using Microsoft Agent Framework
    web_search_agent = ChatAgent(
        name="web_search_agent",
        description="An agent who can search the web to conduct research and answer open questions",
        chat_client=chat_client,
        tools=[get_bing_snippet],
    )
    
    editor_agent = ChatAgent(
        name="editor", 
        description="An expert editor of written articles who can read an article and make suggestions for improvements and additional topics that should be researched",
        chat_client=chat_client,
        instructions="You are an expert editor. You carefully read an article and make suggestions for improvements and suggest additional topics that should be researched to improve the article quality."
    )

    verifier_agent = ChatAgent(
        name="verifier_agent", 
        description="A responsible agent who will verify the facts and ensure that the article is accurate and well-written",
        chat_client=chat_client,
        tools=[get_bing_snippet],
        instructions="You are responsible for ensuring the article's accuracy. You should use the Bing tool to search the internet to verify any relevant facts, and explicitly approve or reject the article based on accuracy, giving your reasoning. You can ask for rewrites if you find inaccuracies."
    )

    writer_assistant = ChatAgent(
        name="writer_assistant", 
        description="A high-quality journalist agent who excels at writing a first draft of an article as well as revising the article based on feedback from the other agents",
        chat_client=chat_client,
        instructions="You are a high-quality journalist agent who excels at writing a first draft of an article as well as revising the article based on feedback from the other agents. Do not just write bullet points on how you would write the article, but actually write it. You can also ask for research to be conducted on certain topics."
    )

    orchestrator_agent = ChatAgent(
        name="orchestrator_agent", 
        description="Team leader who verifies when the article is complete and meets all requirements",
        chat_client=chat_client,
        instructions="You are leading a journalism team that conducts research to craft high-quality articles. You ensure that the output contains an actual well-written article, not just bullet points on what or how to write the article. If the article isn't to that level yet, ask the writer for a rewrite. If the team has written a strong article with a clear point that meets the requirements, and has been reviewed by the editor, and has been fact-checked and approved by the verifier agent, and approved by the user, then reply 'TERMINATE'. Otherwise state what condition has not yet been met."
    )

    # Create a list of all agents for group chat simulation
    agents = [writer_assistant, web_search_agent, editor_agent, verifier_agent, orchestrator_agent]
    
    # Define the task prompt
    task_prompt = "Ask the user to describe the article they want to write. They can include some starting bullet points if they want. Today's date is " + str(datetime.date.today())

    # Initialize console for rich output
    console = Console()
    
    # Start the conversation with the orchestrator
    messages = []
    
    # Orchestrator starts by asking user
    result = await orchestrator_agent.run(task=task_prompt)
    response_text = result.text if hasattr(result, 'text') else str(result)
    
    text = Text()
    text.append(orchestrator_agent.name, style="bold magenta")
    text.append(": ")
    console.print(text)
    md = Markdown(response_text)
    console.print(md)
    
    messages.append({"role": "assistant", "name": orchestrator_agent.name, "content": response_text})
    
    # Get user input
    user_input = input("\nYour response: ")
    messages.append({"role": "user", "content": user_input})
    
    # Run the group chat loop
    conversation_active = True
    max_turns = 30
    turn_count = 0
    last_speaker = None
    
    while conversation_active and turn_count < max_turns:
        turn_count += 1
        
        # Simple selector logic: use orchestrator to decide who speaks next
        # In MAF, we simulate this by having the orchestrator review and select
        selector_prompt = f"""Based on the conversation history, who should speak next to move the article creation forward?
Available agents:
- writer_assistant: Writes and revises the article
- web_search_agent: Searches the web for information
- editor: Reviews and suggests improvements
- verifier_agent: Verifies facts and accuracy
- orchestrator_agent: Coordinates and decides when complete (YOU)
- User: Can provide input

Last speaker: {last_speaker if last_speaker else 'None'}

Reply with ONLY the agent name (or 'User' to ask for user input, or 'TERMINATE' if the article is complete and approved).
"""
        
        # Ask orchestrator to select next speaker
        selector_messages = messages + [{"role": "user", "content": selector_prompt}]
        selector_result = await orchestrator_agent.run(messages=selector_messages)
        next_speaker_text = selector_result.text if hasattr(selector_result, 'text') else str(selector_result)
        next_speaker = next_speaker_text.strip()
        
        # Check for termination
        if "TERMINATE" in next_speaker.upper():
            console.print("\n[bold green]Article creation completed![/bold green]")
            conversation_active = False
            break
        
        # Find the agent
        if "user" in next_speaker.lower():
            # User's turn
            user_response = input("\n[User input]: ")
            messages.append({"role": "user", "content": user_response})
            last_speaker = "User"
            continue
        
        # Find matching agent
        current_agent = None
        for agent in agents:
            if agent.name.lower() in next_speaker.lower():
                current_agent = agent
                break
        
        if not current_agent:
            # Default to writer if selector didn't choose clearly
            current_agent = writer_assistant
        
        try:
            # Run the selected agent
            result = await current_agent.run(messages=messages)
            response_text = result.text if hasattr(result, 'text') else str(result)
            
            # Display the response
            text = Text()
            text.append(current_agent.name, style="bold magenta")
            text.append(": ")
            console.print(text)
            
            md = Markdown(response_text)
            console.print(md)
            
            # Add to conversation history
            messages.append({"role": "assistant", "name": current_agent.name, "content": response_text})
            last_speaker = current_agent.name
            
            # Check if response contains termination signal
            if "TERMINATE" in response_text.upper() and current_agent.name == "orchestrator_agent":
                console.print("\n[bold green]Article creation completed![/bold green]")
                conversation_active = False
                
        except Exception as e:
            console.print(f"[bold red]Error with {current_agent.name}: {e}[/bold red]")
            # Continue with next agent
    
    if turn_count >= max_turns:
        console.print("\n[bold yellow]Maximum turns reached. Ending conversation.[/bold yellow]")



# Load env variables
load_dotenv()
azure_oai_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_model_deployment = os.getenv("AZURE_MODEL_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
bing_endpoint = os.getenv("BING_ENDPOINT")
bing_api_key = os.getenv("BING_API_KEY") 

# Run
asyncio.run(main())
