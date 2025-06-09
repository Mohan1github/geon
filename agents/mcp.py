#!/usr/bin/env python3
"""
AI Agent with FastMCP Server and LLM Tool Selection
This creates an MCP server with various tools and an AI agent that uses an LLM to select appropriate tools.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import httpx
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("AI Agent Server")

class LLMToolSelector:
    """LLM-based tool selector that decides which tools to use for given tasks"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.available_tools = {}
        
    def register_tool(self, name: str, description: str, parameters: dict):
        """Register a tool with the selector"""
        self.available_tools[name] = {
            "description": description,
            "parameters": parameters
        }
    
    async def select_tools(self, user_query: str) -> List[Dict[str, Any]]:
        """Use LLM to select appropriate tools for the user query"""
        
        # Create tool descriptions for the LLM
        tool_descriptions = []
        for name, info in self.available_tools.items():
            tool_descriptions.append(f"- {name}: {info['description']}")
        
        system_prompt = f"""You are a tool selection assistant. Given a user query, select the most appropriate tools to accomplish the task.

Available tools:
{chr(10).join(tool_descriptions)}

Respond with a JSON array of tool selections. Each selection should have:
- "tool": tool name
- "reasoning": why this tool is needed
- "parameters": suggested parameters (can be partial)

Example response:
[
    {{
        "tool": "web_search",
        "reasoning": "Need to search for current information",
        "parameters": {{"query": "relevant search terms"}}
    }}
]

If no tools are needed, return an empty array []."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User query: {user_query}"}
        ]
        
        try:
            if self.api_key:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": 0.1,
                            "max_tokens": 1000
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result["choices"][0]["message"]["content"]
                        return json.loads(content)
            else:
                # Fallback: Simple rule-based selection when no API key
                return self._fallback_tool_selection(user_query)
                
        except Exception as e:
            logger.error(f"Error in LLM tool selection: {e}")
            return self._fallback_tool_selection(user_query)
    
    def _fallback_tool_selection(self, query: str) -> List[Dict[str, Any]]:
        """Fallback rule-based tool selection"""
        selections = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["search", "find", "lookup", "google"]):
            selections.append({
                "tool": "web_search",
                "reasoning": "Query contains search-related keywords",
                "parameters": {"query": query}
            })
        
        if any(word in query_lower for word in ["weather", "temperature", "forecast"]):
            selections.append({
                "tool": "get_weather",
                "reasoning": "Query is about weather information",
                "parameters": {"location": "default"}
            })
        
        if any(word in query_lower for word in ["calculate", "math", "compute"]):
            selections.append({
                "tool": "calculate",
                "reasoning": "Query involves mathematical computation",
                "parameters": {"expression": query}
            })
        
        return selections

# Initialize the LLM tool selector
tool_selector = LLMToolSelector()

# Tool implementations
@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web for information"""
    tool_selector.register_tool("web_search", "Search the web for information", {"query": "string"})
    
    try:
        async with httpx.AsyncClient() as client:
            # Using DuckDuckGo Instant Answer API as an example
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1"}
            )
            
            if response.status_code == 200:
                data = response.json()
                abstract = data.get("Abstract", "")
                if abstract:
                    return f"Search results for '{query}':\n{abstract}"
                else:
                    return f"Search completed for '{query}' but no direct answer found. You may need to search more specifically."
            else:
                return f"Search failed with status code: {response.status_code}"
                
    except Exception as e:
        return f"Search error: {str(e)}"

@mcp.tool()
async def get_weather(location: str = "New York") -> str:
    """Get current weather information for a location"""
    tool_selector.register_tool("get_weather", "Get current weather information", {"location": "string"})
    
    # Mock weather data - in real implementation, use a weather API
    weather_data = {
        "New York": {"temp": "22°C", "condition": "Partly cloudy", "humidity": "65%"},
        "London": {"temp": "18°C", "condition": "Rainy", "humidity": "80%"},
        "Tokyo": {"temp": "25°C", "condition": "Sunny", "humidity": "55%"},
    }
    
    data = weather_data.get(location, weather_data["New York"])
    return f"Weather in {location}: {data['temp']}, {data['condition']}, Humidity: {data['humidity']}"

@mcp.tool()
async def calculate(expression: str) -> str:
    """Perform mathematical calculations"""
    tool_selector.register_tool("calculate", "Perform mathematical calculations", {"expression": "string"})
    
    try:
        # Simple calculator - in production, use a proper math parser
        import re
        
        # Basic safety check
        if re.search(r'[^0-9+\-*/().\s]', expression):
            return "Error: Only basic mathematical operations are allowed"
        
        result = eval(expression)
        return f"Calculation: {expression} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

@mcp.tool()
async def file_operations(operation: str, filename: str, content: str = "") -> str:
    """Perform file operations (read, write, list)"""
    tool_selector.register_tool("file_operations", "Perform file operations", 
                               {"operation": "string", "filename": "string", "content": "string"})
    
    try:
        if operation == "write":
            with open(filename, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {filename}"
        
        elif operation == "read":
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    content = f.read()
                return f"Contents of {filename}:\n{content}"
            else:
                return f"File {filename} not found"
        
        elif operation == "list":
            files = os.listdir("." if not filename else filename)
            return f"Files in directory: {', '.join(files)}"
        
        else:
            return f"Unknown operation: {operation}"
            
    except Exception as e:
        return f"File operation error: {str(e)}"

@mcp.tool()
async def get_system_info() -> str:
    """Get system information"""
    tool_selector.register_tool("get_system_info", "Get system information", {})
    
    import platform
    import psutil
    
    info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Architecture": platform.architecture()[0],
        "Processor": platform.processor(),
        "Memory": f"{psutil.virtual_memory().total // (1024**3)} GB",
        "CPU Cores": psutil.cpu_count(),
        "Current Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return "System Information:\n" + "\n".join([f"{k}: {v}" for k, v in info.items()])

# AI Agent class that orchestrates tool usage
class AIAgent:
    """Main AI Agent that processes queries and uses tools intelligently"""
    
    def __init__(self, tool_selector: LLMToolSelector):
        self.tool_selector = tool_selector
        self.tool_registry = {
            "web_search": web_search,
            "get_weather": get_weather,
            "calculate": calculate,
            "file_operations": file_operations,
            "get_system_info": get_system_info
        }
    
    async def process_query(self, query: str) -> str:
        """Process a user query by selecting and executing appropriate tools"""
        
        logger.info(f"Processing query: {query}")
        
        # Get tool selections from LLM
        selected_tools = await self.tool_selector.select_tools(query)
        
        if not selected_tools:
            return "I don't think any specific tools are needed for this query. Could you be more specific about what you'd like me to help you with?"
        
        results = []
        results.append(f"I'll help you with that! I've selected {len(selected_tools)} tool(s) to address your query:\n")
        
        for selection in selected_tools:
            tool_name = selection["tool"]
            reasoning = selection["reasoning"]
            parameters = selection.get("parameters", {})
            
            results.append(f"\n🔧 Using {tool_name}: {reasoning}")
            
            if tool_name in self.tool_registry:
                try:
                    # Execute the tool
                    tool_func = self.tool_registry[tool_name]
                    result = await tool_func(**parameters)
                    results.append(f"Result: {result}")
                except Exception as e:
                    results.append(f"Error executing {tool_name}: {str(e)}")
            else:
                results.append(f"Tool {tool_name} not found in registry")
        
        return "\n".join(results)

# Initialize the AI agent
ai_agent = AIAgent(tool_selector)

# Main query processing endpoint
@mcp.tool()
async def process_user_query(query: str) -> str:
    """Main entry point for processing user queries with AI agent"""
    return await ai_agent.process_query(query)

# Resource for agent status
@mcp.resource("agent://status")
async def get_agent_status() -> str:
    """Get the current status of the AI agent"""
    available_tools = list(ai_agent.tool_registry.keys())
    return f"""AI Agent Status:
- Status: Active
- Available Tools: {', '.join(available_tools)}
- LLM Model: {tool_selector.model}
- API Key Available: {'Yes' if tool_selector.api_key else 'No (using fallback)'}
- Total Registered Tools: {len(available_tools)}
"""

# Run the server
async def main():
    """Main function to run the MCP server"""
    # Register all tools with the selector
    for tool_name, tool_func in ai_agent.tool_registry.items():
        if hasattr(tool_func, '__doc__'):
            tool_selector.register_tool(tool_name, tool_func.__doc__, {})
    
    logger.info("Starting AI Agent MCP Server...")
    logger.info(f"Available tools: {list(ai_agent.tool_registry.keys())}")
    
    # Run the FastMCP server
    await mcp.run()

if __name__ == "__main__":
    # Example usage
    print("AI Agent with FastMCP Server")
    print("=============================")
    print("This server provides an AI agent that uses LLM to select appropriate tools.")
    print("\nTo run the server:")
    print("python ai_agent_server.py")
    print("\nExample queries to try:")
    print("- 'Search for information about Python programming'")
    print("- 'What's the weather like in Tokyo?'")
    print("- 'Calculate 15 * 23 + 100'")
    print("- 'Show me system information'")
    print("\nSet OPENAI_API_KEY environment variable for LLM-based tool selection.")
    
    asyncio.run(main())