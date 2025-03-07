import asyncio
import datetime
import json
import requests
import os
from dotenv import load_dotenv
from typing import List, Sequence
from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown
from autogen_agentchat.agents import AssistantAgent, BaseChatAgent, UserProxyAgent
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import ChatMessage, StopMessage, TextMessage
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


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
    # Define agents
    user_proxy = UserProxyAgent("User")

    web_search_agent = AssistantAgent(
        name="web_search_agent",
        description="An agent who can search the web to research products and find product prices",
        model_client=AzureOpenAIChatCompletionClient(
            model=azure_model_deployment,
            api_version=azure_api_version,
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            model_capabilities={
                "vision": True,
                "function_calling": True,
                "json_output": True,
            },
        ),
        tools=[get_bing_snippet],
    )

    summarizer_agent = AssistantAgent(
        name = "summarizer_agent", 
        description="A high-quality agent who can summarize the product research and make a strong recommendation on the best purchase.  It can make an initial recommendation, as well as revising the recommendation based on feedback from the other agents",
        model_client=AzureOpenAIChatCompletionClient(
            model=azure_model_deployment,
            api_version=azure_api_version,
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            model_capabilities={
                "vision": True,
                "function_calling": True,
                "json_output": True,
            },
        ), 
        system_message="You are a high-quality agent who excels at summarizing product research and will make a strong recommendation on the best product to buy.  You can make an initial recommendation on product purchases, as well as revising your recommendation based on feedback from the other agents.  You can also ask for research to be conducted on certain products. "
    )

    budget_agent = AssistantAgent(
        name = "budget_assistant", 
        description="A fiscally-responsible agent who will ask for your budget and ensure you stay within it",
        model_client=AzureOpenAIChatCompletionClient(
            model=azure_model_deployment,
            api_version=azure_api_version,
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            model_capabilities={
                "vision": True,
                "function_calling": True,
                "json_output": True,
            },
        ), 
        system_message="You are responsible for ensuring the product stays under budget.  If you don't know the budget, ask the user.  Approximate the cost of the proposed purchase, and approve or reject the plan based on budget, giving your reasoning. "
    )

    orchestrator_agent = AssistantAgent(
        name = "orchestrator_agent", 
        description="Team leader who determines when the product recommendation meets all requirements and the user has approved the purchase",
        model_client=AzureOpenAIChatCompletionClient(
            model=azure_model_deployment,
            api_version=azure_api_version,
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            model_capabilities={
                "vision": True,
                "function_calling": True,
                "json_output": True,
            },
        ), 
        system_message="You are a leading a team that conducts product research and makes recommendations on the best product to purchase.  If the product doesn't meet the requirments, ask for further product research.  If the product recommendation meets the requirements, and has been reviewed by the user, then reply 'TERMINATE'."
    )


    # Define termination condition
    termination = TextMentionTermination("TERMINATE", sources=["orchestrator_agent"])

    # Define a team
    agent_team = SelectorGroupChat(
        [orchestrator_agent, summarizer_agent, budget_agent, user_proxy, web_search_agent], 
        model_client=AzureOpenAIChatCompletionClient(
            model=azure_model_deployment,
            api_version=azure_api_version,
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            model_capabilities={
                "vision": True,
                "function_calling": True,
                "json_output": True,
            },
        ),
        termination_condition=termination
    )

    # Define the task prompt
    task_prompt = "Ask the user to describe what product to research and any requirements they have. They can include some bullet points if they want. Today's date is " + str(datetime.date.today())

    # Run the team and stream messages
    stream = agent_team.run_stream(task=task_prompt)
    console = Console()
    async for response in stream:
        #print(response)
        text = Text()
        if not isinstance(response, TaskResult):
            # Print the agent name in color
            text.append(response.source, style="bold magenta")
            text.append(": ")
            console.print(text)
            if isinstance(response, str):
                md = Markdown(response.content)
                console.print(md)
            else:
                console.print(response.content)
        else:
            console.print(response.stop_reason)



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
