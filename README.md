# Multiagent Research
An exploration of using multiple agents collaborating to perform research

## What is changed when compared to the original repo? 
This fork contains a change from Bing v7 which is not available anymore to the new Bing Grounding API. The original repo used the v7 API which is now deprecated. The new API is a bit different, but the functionality is similar. Also, as the new Bing Grounding API can only be used via the AI Agent Service the code has been modified to use the Azure AI Foundry service. Also, as AI Agent Service is used as alternative to AutoGen as a result. 

## Scenarios
This repo contains two different scenarios, both implemented using multiple agents collaborating using AI Agent Service. 
+ **journalism_research.py** - run this script for a group of agents to conduct online research to craft a news article
+ **shopping.py** - run this script for a group of agents to conduct online research to compare products and make a recommendation

## Setup
You will first need to create an [Azure AI Foundry](https://portal.azure.com/#create/Microsoft.CognitiveServicesOpenAI) resource with a hub and project. Copy the project connection string and update the .env file with it.

To use the Bing Grounding API, go to "Connected Resources" in the project blade and add a connection to the Bing resource. Fetch the name of the Bing resource and copy/paste it into the .env file as well. Make sure you have deployed a Grounding with Bing resource in Azure first. 

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
