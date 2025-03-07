import asyncio
import datetime
import requests
import os
from dotenv import load_dotenv
from typing import List, Sequence
from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown
from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import ChatMessage, StopMessage, TextMessage
from autogen_agentchat.task import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_core.base import CancellationToken
from autogen_core.components.tools import FunctionTool
from autogen_ext.models import AzureOpenAIChatCompletionClient

class UserProxyAgent(BaseChatAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name, "A human user.")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage, StopMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        user_input = await asyncio.get_event_loop().run_in_executor(None, input, "Enter your response: ")
        if "TERMINATE" in user_input:
            return Response(chat_message=StopMessage(content="User has terminated the conversation.", source=self.name))
        return Response(chat_message=TextMessage(content=user_input, source=self.name))

    async def reset(self, cancellation_token: CancellationToken) -> None:
        pass


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
        return tuple(results_list)
    else:
        error = f"Error: {response.status_code} - {response.text}"
        print(error)
        return error


async def main() -> None:
    # Define agents
    user_proxy = UserProxyAgent("User")

    web_search_agent = AssistantAgent(
        name="web_search_agent",
        description="An agent who can search the web to conduct research and answer open questions",
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
    
    editor_agent = AssistantAgent(
        name = "editor", 
        description="An expert editor of written articles who can read an article and make suggestions for improvements and additional topics that should be researched",
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
        system_message="You are an expert editor.  You carefully read an article and make suggestions for improvements and suggest additional topics that should be researched to improve the article quality."
    )

    verifier_agent = AssistantAgent(
        name = "verifier_agent", 
        description="A responsible agent who will verify the facts and ensure that the article is accurate and well-written",
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
        system_message="You are responsible for ensuring the article's accuracy.  You should use the Bing tool to search the internet to verify any relevant facts, and explicitly approve or reject the article based on accuracy, giving your reasoning. You can ask for rewrites if you find inaccuracies."
    )

    writer_assistant = AssistantAgent(
        name = "writer_assistant", 
        description="A high-quality journalist agent who excels at writing a first draft of an article as well as revising the article based on feedback from the other agents",
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
        system_message="You are a high-quality journalist agent who excels at writing a first draft of an article as well as revising the article based on feedback from the other agents.  You can also ask for research to be conducted on certain topics. "
    )

    orchestrator_agent = AssistantAgent(
        name = "orchestrator_agent", 
        description="Team leader who verifies when the article is complete and meets all requirements",
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
        system_message="You are a leading a journalism team that conducts research to craft high-quality articles.  You ensure that the output contains an actual well-written article, not just bullet points on what or how to write the article.  If the article isn't to that level yet, ask the writer for a rewrite.  If the team has written a strong article with a clear point that meets the requirements, and has been reviewed by the editor, and has been fact-checked and approved by the verifier agent, and approved by the user, then reply 'TERMINATE'.  Otherwise state what condition has not yet been met."
    )


    # Define termination condition
    termination = TextMentionTermination("TERMINATE")

    # Define a team
    agent_team = SelectorGroupChat(
        [orchestrator_agent, writer_assistant, editor_agent, verifier_agent, user_proxy, web_search_agent], 
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
    task_prompt = "Ask the user to describe the article they want to write. They can include some starting bullet points if they want. Today's date is " + str(datetime.date.today())

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
