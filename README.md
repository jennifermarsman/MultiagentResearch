# Multiagent Research
An exploration of using multiple agents collaborating to perform research

## Scenarios
This repo contains two different scenarios implemented using multiple agents collaborating with **Microsoft Agent Framework**:
+ **journalism_research.py** - run this script for a group of agents to conduct online research to craft a news article
+ **shopping.py** - run this script for a group of agents to conduct online research to compare products and make a recommendation

## Setup
You will first need to create an [Azure OpenAI resource](https://portal.azure.com/#create/Microsoft.CognitiveServicesOpenAI) with a GPT-4o model deployment, and update the .env file with the endpoint and key (and the deployment name if you change from the default).  

To use the Bing Search API, you will also need to create a [Bing resource](https://portal.azure.com/#create/Microsoft.BingSearch) and update the .env file with its key.  

Finally, use the following commands in a python environment (such as an Anaconda prompt window) to set up your environment. This creates and activates an environment and installs the required packages. For subsequent runs after the initial install, you will only need to activate the environment and then run the python script.

### First run
```
conda create --name research -y
conda activate research

pip install -r requirements.txt
python journalism_research.py
```

### Subsequent runs
```
conda activate research
python journalism_research.py
```

## Migration to Microsoft Agent Framework
This codebase has been migrated from AutoGen 0.4 to Microsoft Agent Framework. Key changes include:
- Updated from `autogen-agentchat` to `agent-framework` packages
- Agents now use `ChatAgent` from Microsoft Agent Framework
- Azure OpenAI integration via `AzureOpenAIChatClient`
- Custom group chat orchestration (MAF doesn't have direct SelectorGroupChat equivalent)
- Simplified authentication with Azure CLI or API key fallback

### Authentication
The code supports two authentication methods:
1. **Azure CLI (Recommended)**: Run `az login` before executing the script
2. **API Key**: Set `AZURE_OPENAI_API_KEY` in your .env file
