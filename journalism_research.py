import asyncio
import datetime
import requests
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown
from bing_skill import BingSearchSkill

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import KernelFunctionSelectionStrategy
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import KernelFunctionTerminationStrategy
from semantic_kernel.contents.history_reducer.chat_history_truncation_reducer import ChatHistoryTruncationReducer
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

# TODO: IS the following used?
from semantic_kernel.agents.strategies.termination.default_termination_strategy import DefaultTerminationStrategy


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


def create_kernel() -> Kernel:
    kernel = Kernel()
    # Configure the AzureChatCompletion service using env vars.
    azure_service = AzureChatCompletion(
        deployment_name=azure_model_deployment,
        api_version=azure_api_version,
        endpoint=azure_oai_endpoint,
        api_key=azure_oai_key,
    )
    kernel.add_service(azure_service)
    return kernel

def create_bing_kernel() -> Kernel:
    kernel = Kernel()
    azure_service = AzureChatCompletion(
        deployment_name=azure_model_deployment,
        api_version=azure_api_version,
        endpoint=azure_oai_endpoint,
        api_key=azure_oai_key,
    )
    kernel.add_service(azure_service)

    # Register your Bing search skill
    bing_skill = BingSearchSkill()
    kernel.add_plugin(bing_skill, plugin_name="BingSearch")

    # Use the pre-built Bing search skill
    # TODO: see https://learn.microsoft.com/en-us/semantic-kernel/concepts/text-search/out-of-the-box-textsearch/bing-textsearch?pivots=programming-language-python

    return kernel


async def main() -> None:
    # Load configuration from environment variables.
    load_dotenv()
    global azure_oai_endpoint, azure_oai_key, azure_model_deployment, azure_api_version, bing_endpoint, bing_api_key
    azure_oai_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")
    azure_oai_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_model_deployment = os.getenv("AZURE_MODEL_DEPLOYMENT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    bing_endpoint = os.getenv("BING_ENDPOINT")
    bing_api_key = os.getenv("BING_API_KEY")

    # Create a semantic kernel instance.  
    kernel = create_kernel()  

    # Define agent names (and matching service_ids)  
    ORCHESTRATOR = "orchestrator_agent"  
    WRITER = "writer_assistant"  
    EDITOR = "editor_agent"  
    VERIFIER = "verifier_agent"  
    WEB_SEARCH = "web_search_agent" 

    # Define agents
    #WEB_SEARCH_NAME = "web_search_agent"
    #REVIEWER_NAME = "reviewer_agent"
    #ORCHESTRATOR_NAME = "orchestrator_agent"
    #WRITER_NAME = "writer_assistant"
    #EDITOR_NAME = "editor_agent"
    #VERIFIER_NAME = "verifier_agent"
    #USER_NAME = "user_proxy"

    '''
    user_proxy = UserProxyAgent("User")

    agent_reviewer = ChatCompletionAgent(
        service_id=WEB_SEARCH_NAME,
        kernel=kernel,
        name=WEB_SEARCH_NAME,
        instructions="""
        Your responsibility is to review and identify how to improve user provided content.
        If the user has provided input or direction for content already provided, specify how to address this input.
        Never directly perform the correction or provide an example.
        Once the content has been updated in a subsequent response, review it again until it is satisfactory.

        RULES:
        - Only identify suggestions that are specific and actionable.
        - Verify previous suggestions have been addressed.
        - Never repeat previous suggestions.
        """,
    )
    '''
    # Open questions:
    # 1. How to create a user proxy agent?
    # 2. Non round robin selection strategy?
    # 3. Does skill have to be attached to kernel?  Separate kernels for non-skill agents?

    # Create ChatCompletionAgent instances for each role. 

    web_search_agent = ChatCompletionAgent(
        service_id=WEB_SEARCH,
        kernel=create_bing_kernel(),
        name=WEB_SEARCH,
        instructions="""  
            You are a web search agent. You have access to Bing search (via the BingSearch.search function) to retrieve relevant
            information to conduct research, to fact-check, and to answer open questions. Provide concise search results.
            """,
        #instructions="You are an agent who can search the web to conduct research and answer open questions.",
        #functions=[get_bing_snippet],
        description="An agent who can search the web to conduct research and answer open questions",
        #skills=[get_bing_snippet],
    )

    editor_agent = ChatCompletionAgent(  
        service_id=EDITOR,  
        kernel=kernel,  
        name=EDITOR,  
        instructions="""  
        You are an expert editor of written articles. Read the article carefully and suggest actionable improvements
        and additional topics that should be researched to improve the article's quality.
        """,
        # old system_message="You are an expert editor.  You carefully read an article and make suggestions for improvements and suggest additional topics that should be researched to improve the article quality."
    )

    verifier_agent = ChatCompletionAgent(  
        service_id=VERIFIER,  
        kernel=create_bing_kernel(),  
        name=VERIFIER,  
        instructions="""  
            You are responsible for ensuring the article's accuracy.  Verify that the article is factually correct and well-written. Use available research
            tools if needed and ask for rewrites if inaccuracies are found.
            """,
        # tools=[get_bing_snippet],
        # old system_message="You are responsible for ensuring the article's accuracy.  You can search the internet to verify any relevant facts, and explicitly approve or reject the article based on accuracy, giving your reasoning. You can ask for rewrites if you find inaccuracies."
    )

    writer_assistant = ChatCompletionAgent(  
        service_id=WRITER,  
        kernel=kernel,  
        name=WRITER,  
        instructions="""  
        You are a high-quality journalist agent who excels at writing a first draft of an article as well as revising it
        based on feedback. Follow the directions provided by your teammates and produce a well-written article.
        """,
        # old system_message="You are a high-quality journalist agent who excels at writing a first draft of an article as well as revising the article based on feedback from the other agents.  You can also ask for research to be conducted on certain topics. "
    )

    orchestrator_agent = ChatCompletionAgent(  
        service_id=ORCHESTRATOR,  
        kernel=kernel,  
        name=ORCHESTRATOR,  
        instructions="""  
        You are leading a journalism team that conducts research to craft high-quality, well-written articles.
        Ensure that the final article meets rigorous standards. If the article has been reviewed by your teammates
        (the editor and verifier) and is complete, reply with "TERMINATE". Otherwise, instruct the next agent accordingly.
        """,
        # old system_message="You are a leading a journalism team that conducts research to craft high-quality articles.  You ensure that the output contains an actual well-written article, not just bullet points on what or how to write the article.  If the article isn't to that level yet, ask the writer for a rewrite.  If the team has written a strong article with a clear point that meets the requirements, and has been reviewed by the editor, and has been fact-checked and approved by the verifier agent, then reply 'TERMINATE'."
    )


    # Define a selection function (using a Kernel prompt) to choose the next speaker.  
    selection_function = KernelFunctionFromPrompt(  
        function_name="selection",  
        prompt=f"""  
        Examine the provided RESPONSE and choose the next participant.
        State only the name of the chosen participant without explanation.
        Never choose the participant named in the RESPONSE.

        Choose only from these participants:

        {WRITER}

        {WEB_SEARCH}

        {EDITOR}

        {VERIFIER}

        {ORCHESTRATOR}

        Rules:

        Based on the content of the RESPONSE and the current needs in the conversation, choose the next participant. 

        Otherwise follow a round-robin order starting from {WRITER}.  The {ORCHESTRATOR} is responsible for ending the conversation and should go last.

        RESPONSE:
        {{{{$lastmessage}}}}
        """
    )
    # Removed from AI-generated selection function:
    # If the RESPONSE is from user input, select {ORCHESTRATOR}.
    # Otherwise follow a round-robin order starting from {ORCHESTRATOR}.

    # TODO: do we still need the orchestrator or is the termination function enough?  Can it ask for changes and be heard by the other agents?
    
    
    # Define a termination function to end the conversation when the article is complete
    termination_keyword = "terminate" # we will check using lower-case
    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=f"""
        Examine the RESPONSE and determine whether the article is complete and meets all requirements.
        + Ensure that the proper research has been conducted to craft a high-quality article.  
        + Ensure that the output contains an actual well-written article, not just bullet points on what or how to write the article.  If the article isn't to that level yet, ask the writer for a rewrite.  
        If the team has written a strong article with a clear point that meets the requirements, and has been reviewed by the editor, and has been fact-checked and approved by the verifier agent, and has been approved by the user, then respond with a single word: 'TERMINATE'.
        If further revisions are needed, do not respond with that word.

        RESPONSE:
        {{{{$lastmessage}}}}
        """
    )

    #history_reducer = ChatHistoryTruncationReducer(max_tokens=3000)
    history_reducer = ChatHistoryTruncationReducer(target_count=5)

    print("About to start chat")

    # Define a team
    chat = AgentGroupChat(
        agents=[orchestrator_agent, writer_assistant, editor_agent, verifier_agent, web_search_agent],
        selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=orchestrator_agent,
            function=selection_function,
            kernel=kernel,
            result_parser=lambda result: str(result.value[0]).strip() if result.value and result.value[0] is not None else WRITER,
            history_variable_name="lastmessage",
            history_reducer=history_reducer,
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[orchestrator_agent],
            function=termination_function,
            kernel=kernel,
            result_parser=lambda result: termination_keyword in str(result.value[0]).lower() if result.value and result.value[0] is not None else False,
            history_variable_name="lastmessage",
            maximum_iterations=10,
            history_reducer=history_reducer,
        ),
    )

    # Prepare the initial task prompt. In this case we ask the user to describe the article.
    task_prompt = "Please describe the article you want to write. You may include some bullet points. Today's date is " + str(datetime.date.today())

    # Add the task prompt as a user message to the chat history.
    await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=task_prompt))

    # Create a rich console to stream conversation results.
    console = Console()
    console.print("[bold green]Agent Team Chat Starting…[/bold green]\n")

    # Run the AgentGroupChat; responses will be streamed back asynchronously.
    try:
        async for response in chat.invoke():
            # Continue only if a valid response was received.
            if not response or not response.name:
                continue
            # Print the agent name and its response.  
            console.print(f"[bold magenta]{response.name}[/bold magenta]:")  
            # Render content as Markdown (if applicable) so that formatting is preserved.  
            console.print(Markdown(response.content))  
    except Exception as e:
        console.print(f"[red]Error during chat invocation: {e}[/red]")
        console.print("\n[bold green]Conversation terminated.[/bold green]")

asyncio.run(main())

