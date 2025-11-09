# Migration Guide: AutoGen 0.4 → Microsoft Agent Framework

This document explains the migration from AutoGen 0.4 to Microsoft Agent Framework performed on this repository.

## Overview

**Microsoft Agent Framework** is the successor to AutoGen, combining the best of AutoGen's multi-agent orchestration with enhanced enterprise features. AutoGen is now in maintenance mode (security fixes only), while Microsoft Agent Framework is actively developed.

## What Changed

### 1. Dependencies (requirements.txt)

**Before:**
```
autogen-agentchat==0.4.8
autogen-ext[openai,azure,docker]==0.4.8
python-dotenv
rich
```

**After:**
```
agent-framework
azure-identity
python-dotenv
rich
requests
```

### 2. Imports

**Before:**
```python
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
```

**After:**
```python
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential
```

### 3. Agent Creation

**Before:**
```python
agent = AssistantAgent(
    name="writer",
    description="...",
    model_client=AzureOpenAIChatCompletionClient(
        model=model_name,
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=key,
        model_capabilities={...}
    ),
    system_message="...",
    tools=[...]
)
```

**After:**
```python
chat_client = AzureOpenAIChatClient(
    endpoint=endpoint,
    model=model_name,
    api_version=api_version,
    api_key=key,
    # OR use Azure CLI credentials:
    # credential=DefaultAzureCredential()
)

agent = ChatAgent(
    name="writer",
    description="...",
    chat_client=chat_client,
    instructions="...",  # replaces system_message
    tools=[...]
)
```

### 4. Group Chat Orchestration

**Before (AutoGen):**
```python
team = SelectorGroupChat(
    agents=[agent1, agent2, ...],
    model_client=model_client,
    termination_condition=TextMentionTermination("TERMINATE")
)

stream = team.run_stream(task=prompt)
async for response in stream:
    # Handle response
```

**After (Microsoft Agent Framework):**
```python
# Custom orchestration loop
messages = []
while conversation_active:
    # Orchestrator selects next speaker
    selector_prompt = "Based on conversation, who should speak next?"
    result = await orchestrator_agent.run(messages=messages + [selector_prompt])
    next_speaker = result.text
    
    # Find and run the selected agent
    current_agent = find_agent(next_speaker)
    result = await current_agent.run(messages=messages)
    messages.append({"role": "assistant", "name": current_agent.name, "content": result.text})
    
    # Check for termination
    if "TERMINATE" in result.text:
        break
```

## Key Differences

### Architecture

| Feature | AutoGen 0.4 | Microsoft Agent Framework |
|---------|-------------|---------------------------|
| Group Chat | Built-in `SelectorGroupChat` | Custom orchestration required |
| Agent Types | `AssistantAgent`, `UserProxyAgent` | `ChatAgent` |
| Model Client | `AzureOpenAIChatCompletionClient` | `AzureOpenAIChatClient` |
| Instructions | `system_message` parameter | `instructions` parameter |
| Authentication | API key only | API key or Azure CLI credentials |
| Message Handling | Event-driven streaming | Message list with run() method |

### Benefits of Microsoft Agent Framework

1. **Active Development**: New features and improvements ongoing
2. **Better Authentication**: Support for Azure CLI and managed identities
3. **Unified Framework**: Combines AutoGen and Semantic Kernel concepts
4. **Enterprise Ready**: Enhanced observability, security, and governance
5. **Multi-language**: Python and .NET support

## Migration Checklist

- [x] Update requirements.txt with new packages
- [x] Replace AutoGen imports with Microsoft Agent Framework imports
- [x] Convert `AssistantAgent` → `ChatAgent`
- [x] Update model client initialization
- [x] Replace `system_message` with `instructions`
- [x] Implement custom group chat orchestration
- [x] Remove `UserProxyAgent` (handle user input directly)
- [x] Update termination logic
- [x] Test authentication (Azure CLI or API key)
- [x] Validate functionality

## Testing

Run the validation script to ensure migration was successful:

```bash
python test_migration.py
```

This validates:
- ✅ All imports work correctly
- ✅ Script syntax is valid
- ✅ Tool functions are properly defined
- ✅ Environment variables are configured

## Running the Migrated Code

### Prerequisites

1. Azure OpenAI resource with GPT-4o deployment
2. Bing Search API key (optional, for web search)
3. Python 3.8+

### Installation

```bash
# Create and activate environment
conda create --name research -y
conda activate research

# Install dependencies
pip install -r requirements.txt

# Authenticate with Azure CLI (recommended)
az login

# OR set API key in .env file
# AZURE_OPENAI_API_KEY=your_key_here
```

### Run

```bash
# Journalism research
python journalism_research.py

# Product shopping research  
python shopping.py
```

## Troubleshooting

### Import Errors

If you get import errors, ensure agent-framework is installed:
```bash
pip install agent-framework --upgrade
```

### Authentication Issues

**Azure CLI Method (Recommended):**
```bash
az login
az account set --subscription "your-subscription-id"
```

**API Key Method:**
Add to .env file:
```
AZURE_OPENAI_API_KEY=your_key_here
```

### Agent Not Responding

The custom orchestration loop may need tuning. Check:
- Orchestrator agent instructions are clear
- Agent names match exactly in selector logic
- Message history is maintained properly

## Resources

- [Microsoft Agent Framework Docs](https://learn.microsoft.com/en-us/agent-framework/)
- [AutoGen to MAF Migration Guide](https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Agent Framework GitHub](https://github.com/microsoft/agent-framework)

## Support

For issues with:
- **Migration**: Check this guide and test_migration.py
- **Microsoft Agent Framework**: See official docs
- **Azure OpenAI**: Check Azure portal and service documentation

---

**Migration completed successfully! 🎉**
