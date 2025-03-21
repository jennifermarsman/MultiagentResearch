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

'''
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
'''

async def main() -> None:

    # Load configuration from environment variables.  NOTE: Semantic Kernel uses Pydantic settings, so this is not needed, but we do it for clarity.
    load_dotenv()
    global azure_oai_endpoint, azure_oai_key, azure_model_deployment, azure_api_version, bing_endpoint, bing_api_key
    azure_oai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_oai_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_model_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    bing_endpoint = os.getenv("BING_ENDPOINT")
    bing_api_key = os.getenv("BING_API_KEY")

    # Create a semantic kernel instance.  
    kernel = create_kernel()
    
    # Create an Azure OpenAI service instance.  In this example, I am using the same service for each agent, but you can use different services per agent if needed.
    azure_service = AzureChatCompletion(
        deployment_name=azure_model_deployment,
        api_version=azure_api_version,
        endpoint=azure_oai_endpoint,
        api_key=azure_oai_key,
    )

    # Define agent names 
    ORCHESTRATOR = "orchestrator_agent"  
    WRITER = "writer_assistant"  
    EDITOR = "editor_agent"  
    VERIFIER = "verifier_agent"  
    WEB_SEARCH = "web_search_agent" 

    # Open questions:
    # 1. How to create a user proxy agent?
    # 2. Non round robin selection strategy?
    # 3. Does skill have to be attached to kernel?  Separate kernels for non-skill agents?

    # Create ChatCompletionAgent instances for each role. 

    web_search_agent = ChatCompletionAgent(
        service=azure_service,
        name=WEB_SEARCH,
        instructions="""  
            You are a web search agent. You can issue queries to Bing search (via the BingSearchSkill.search function) to retrieve relevant
            information to conduct research, to fact-check, and to answer open questions. Provide concise search results.
            """,
        description="An agent who can search the web to conduct research and answer open questions",
        plugins=[BingSearchSkill()],
    )

    editor_agent = ChatCompletionAgent(  
        service=azure_service, 
        name=EDITOR,  
        instructions="""  
        You are an expert editor of written articles. Read the article carefully and suggest actionable improvements
        and additional topics that should be researched to improve the article's quality.
        """,
    )

    verifier_agent = ChatCompletionAgent(  
        service=azure_service,  
        name=VERIFIER,  
        instructions="""  
            You are responsible for ensuring the article's accuracy. You can search the internet using the BingSearchSkill.search tool to verify 
            any relevant facts, and explicitly approve or reject the article based on accuracy, giving your reasoning. You should ask for rewrites 
            if you find inaccuracies.
            """,
        plugins=[BingSearchSkill()],
    )

    writer_assistant = ChatCompletionAgent(  
        service=azure_service,  
        name=WRITER,  
        instructions="""  
        You are a high-quality journalist agent who excels at writing a first draft of an article as well as revising it
        based on feedback. Follow the directions provided by your teammates and produce a well-written article. You can also 
        ask for research to be conducted on certain topics if needed.
        """,
    )

    orchestrator_agent = ChatCompletionAgent(  
        service=azure_service,  
        name=ORCHESTRATOR,  
        instructions="""  
        You are leading a journalism team that conducts research to craft high-quality, well-written articles.
        Ensure that the output contains an actual well-written article, not just bullet points on what or how to write the article.  If 
        the article isn't to that level yet, ask the writer for a rewrite.  
        If the team has written a strong article with a clear point that meets the requirements, and has been reviewed by the editor, 
        and has been fact-checked and approved by the verifier agent, then reply 'TERMINATE'.
        Otherwise, state what of the above requirements is missing or could be improved to instruct the next agent accordingly.
        """,
    )


    # Define a selection function (using a Kernel prompt) to choose the next speaker.  
    selection_function = KernelFunctionFromPrompt(  
        function_name="selection",
        prompt=f"""  
        Examine the provided RESPONSE and choose the next participant.
        State only the name of the chosen participant without explanation.
        Never choose the participant who has just spoken named in the RESPONSE.

        Choose only from these participants:

        {WRITER} - high-quality journalist agent who excels at writing a first draft of an article as well as revising the article based on feedback from the other agents

        {WEB_SEARCH} - agent who can search the web to conduct research and answer open questions

        {EDITOR} - expert editor of written articles who reads the article carefully and suggests actionable improvements to improve the article's quality

        {VERIFIER} - agent that is responsible for ensuring the article's accuracy who can search the internet to verify any relevant facts, and approve or reject the article

        {ORCHESTRATOR} - agent who monitors progress of the team and determines when the article is complete

        Rules:

        Based on the content of the RESPONSE and the current needs in the conversation, choose the next participant. 

        Otherwise follow a round-robin order starting from {WRITER}.  The {ORCHESTRATOR} is responsible for ending the conversation and should go last.

        RESPONSE:
        {{{{$lastmessage}}}}
        """
    )
    
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
            initial_agent=writer_assistant,
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

    # Add the task prompt as a message to the chat history.
    await chat.add_chat_message(ChatMessageContent(role=AuthorRole.ASSISTANT, content=task_prompt))

    # Get the user's input for the article description.
    user_input = input(task_prompt + "\n> ").strip()
    await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))

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
