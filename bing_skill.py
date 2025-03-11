import asyncio
import requests
import os
from dotenv import load_dotenv
from semantic_kernel.functions.kernel_function_decorator import kernel_function


class BingSearchSkill:
    @kernel_function(description="Search Bing for a query and return search results", name="search")
    async def search(self, query: str) -> str:
        load_dotenv()
        bing_endpoint = os.getenv("BING_ENDPOINT")
        bing_api_key = os.getenv("BING_API_KEY")

        # Perform a web search using the Bing Web Search API.
        count = 3
        params = {
            'q': query,
            'count': count,
        }
        headers = {'Ocp-Apim-Subscription-Key': bing_api_key} 
        # Note: Using asyncio.to_thread to run the synchronous requests.get
        response = await asyncio.to_thread(requests.get, bing_endpoint, params=params, headers=headers)
        #response = requests.get(bing_endpoint, params=params, headers=headers)

        # Check if the request was successful (HTTP status code 200).
        if response.status_code == 200:
            search_results = response.json()
            results_list = []
            for result in search_results.get('webPages', {}).get('value', []):
                results_list.append(f"Title: {result['name']}, Snippet: {result['snippet']}, URL: {result['url']}")
            return "\n".join(results_list)
        else:
            error_message = f"Error: {response.status_code} - {response.text}"
            print(error_message)
            return error_message
    


'''
# Original version:
    
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

'''

''' 
This makes the skill “BingSearch” available to your agents. (Note: The method is available under the name “search” since you set name="search" above.)

────────────────────────────
3. Instruct your Agent to use the Bing search skill

In SK the agents are provided instructions about their role. If you want one of your agents (or a specialized “web_search” agent) to call the Bing search skill, you have two options:

• Include in its system prompt instructions such as “if you need to verify a fact or look something up, call BingSearch.search with the appropriate query.”
• Use a Kernel selection strategy or even a structured function call (a mechanism that SK supports) to “pull out” the function invocation from the text that the agent produces.

For example, you could give your web search agent these instructions:
web_search_agent = ChatCompletionAgent(
service_id="web_search_agent",
kernel=kernel,
name="web_search_agent",
instructions="""
You are a web search agent. When you need to look up additional information, use the BingSearch.search function.
For example, if you see a statement such as "search: ", then call BingSearch.search with that query.
Return only the search results.
"""
)
'''