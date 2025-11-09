"""
Test script to validate Microsoft Agent Framework migration.
This script tests imports and basic agent creation without requiring Azure credentials.
"""

import sys
import os

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        from agent_framework import ChatAgent
        print("  ✓ ChatAgent import successful")
    except ImportError as e:
        print(f"  ✗ Failed to import ChatAgent: {e}")
        return False
    
    try:
        from agent_framework.azure import AzureOpenAIChatClient
        print("  ✓ AzureOpenAIChatClient import successful")
    except ImportError as e:
        print(f"  ✗ Failed to import AzureOpenAIChatClient: {e}")
        return False
    
    try:
        from azure.identity import DefaultAzureCredential
        print("  ✓ DefaultAzureCredential import successful")
    except ImportError as e:
        print(f"  ✗ Failed to import DefaultAzureCredential: {e}")
        return False
    
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        print("  ✓ Rich library imports successful")
    except ImportError as e:
        print(f"  ✗ Failed to import from rich: {e}")
        return False
    
    return True

def test_script_syntax():
    """Test that both main scripts have valid Python syntax."""
    print("\nTesting script syntax...")
    import ast
    
    scripts = ['journalism_research.py', 'shopping.py']
    for script in scripts:
        try:
            with open(script, 'r') as f:
                ast.parse(f.read())
            print(f"  ✓ {script} has valid syntax")
        except SyntaxError as e:
            print(f"  ✗ {script} has syntax error: {e}")
            return False
    
    return True

def test_tool_function():
    """Test that the Bing search tool function is defined correctly."""
    print("\nTesting tool functions...")
    
    try:
        # Read and parse the script without executing it
        import ast
        with open('journalism_research.py', 'r') as f:
            tree = ast.parse(f.read())
        
        # Look for the get_bing_snippet function
        found_function = False
        is_async = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == 'get_bing_snippet':
                found_function = True
                is_async = True
                break
            elif isinstance(node, ast.FunctionDef) and node.name == 'get_bing_snippet':
                found_function = True
                break
        
        if not found_function:
            print("  ✗ get_bing_snippet function not found in journalism_research.py")
            return False
        
        if not is_async:
            print("  ✗ get_bing_snippet is not an async function")
            return False
        
        print("  ✓ get_bing_snippet function is correctly defined as async")
        return True
    except Exception as e:
        print(f"  ✗ Error testing tool function: {e}")
        return False

def test_environment_variables():
    """Test that environment variable loading works."""
    print("\nTesting environment configuration...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        'AZURE_OPENAI_API_ENDPOINT',
        'AZURE_MODEL_DEPLOYMENT',
        'AZURE_OPENAI_API_VERSION',
        'BING_ENDPOINT',
        'BING_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"  ⚠ Missing environment variables: {', '.join(missing_vars)}")
        print("  ℹ These are required to run the scripts but not needed for testing")
    else:
        print("  ✓ All required environment variables are set")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Microsoft Agent Framework Migration - Validation Tests")
    print("=" * 60)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_script_syntax():
        all_passed = False
    
    if not test_tool_function():
        all_passed = False
    
    if not test_environment_variables():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All validation tests passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some validation tests failed")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
