# Multiagent Research
An exploration of using multiple agents collaborating to perform research

## Scenarios
This repo contains two different scenarios, both implemented using multiple agents collaborating using AutoGen 0.4.  
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
