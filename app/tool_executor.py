"""
Tool Execution Engine
Handles execution of tools/functions requested by the model
"""

import json
import importlib
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import math
import os
import subprocess
import requests
from pathlib import Path


class ToolExecutor:
    """Executes tools/functions requested by the model"""
    
    def __init__(self):
        self.tool_registry: Dict[str, Callable] = {}
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register built-in tools"""
        self.register_tool("get_current_time", self._get_current_time)
        self.register_tool("calculate", self._calculate)
        self.register_tool("get_weather", self._get_weather)
        self.register_tool("read_file", self._read_file)
        self.register_tool("write_file", self._write_file)
        self.register_tool("list_directory", self._list_directory)
    
    def register_tool(self, name: str, func: Callable):
        """Register a tool function"""
        self.tool_registry[name] = func
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with given arguments"""
        if tool_name not in self.tool_registry:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        try:
            func = self.tool_registry[tool_name]
            result = func(**arguments)
            return result
        except Exception as e:
            raise Exception(f"Error executing tool '{tool_name}': {str(e)}")
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for all registered tools"""
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current date and time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "format": {
                                "type": "string",
                                "description": "Time format: 'iso', 'unix', or 'readable'",
                                "enum": ["iso", "unix", "readable"]
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name or location"
                            },
                            "units": {
                                "type": "string",
                                "description": "Temperature units: 'celsius' or 'fahrenheit'",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and directories in a path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory_path": {
                                "type": "string",
                                "description": "Path to the directory to list"
                            }
                        },
                        "required": ["directory_path"]
                    }
                }
            }
        ]
        return schemas
    
    # Built-in tool implementations
    def _get_current_time(self, format: str = "readable") -> str:
        """Get current time"""
        now = datetime.now()
        if format == "iso":
            return now.isoformat()
        elif format == "unix":
            return str(int(now.timestamp()))
        else:  # readable
            return now.strftime("%Y-%m-%d %H:%M:%S")
    
    def _calculate(self, expression: str) -> float:
        """Safely evaluate a mathematical expression"""
        # Only allow safe math operations
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        })
        
        try:
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return float(result)
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")
    
    def _get_weather(self, location: str, units: str = "celsius") -> str:
        """Get weather information (mock implementation)"""
        # This is a mock implementation
        # In production, you would call a real weather API
        return f"Weather in {location}: 22°C, sunny (mock data)"
    
    def _read_file(self, file_path: str) -> str:
        """Read file contents"""
        try:
            # Security: prevent path traversal
            abs_path = os.path.abspath(file_path)
            # Get workspace from environment or use current directory
            workspace_env = os.getenv("WORKSPACE_PATH")
            if workspace_env:
                workspace = os.path.abspath(workspace_env)
            else:
                # Use the project root (parent of app directory)
                workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            
            if not abs_path.startswith(workspace):
                raise ValueError(f"Access denied: File outside workspace")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
    
    def _write_file(self, file_path: str, content: str) -> str:
        """Write file contents"""
        try:
            # Security: prevent path traversal
            abs_path = os.path.abspath(file_path)
            workspace_env = os.getenv("WORKSPACE_PATH")
            if workspace_env:
                workspace = os.path.abspath(workspace_env)
            else:
                # Use the project root (parent of app directory)
                workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            
            if not abs_path.startswith(workspace):
                raise ValueError(f"Access denied: File outside workspace")
            
            # Create directory if needed
            dir_path = os.path.dirname(abs_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            raise ValueError(f"Error writing file: {str(e)}")
    
    def _list_directory(self, directory_path: str) -> List[str]:
        """List directory contents"""
        try:
            # Security: prevent path traversal
            abs_path = os.path.abspath(directory_path)
            workspace_env = os.getenv("WORKSPACE_PATH")
            if workspace_env:
                workspace = os.path.abspath(workspace_env)
            else:
                # Use the project root (parent of app directory)
                workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            
            if not abs_path.startswith(workspace):
                raise ValueError(f"Access denied: Directory outside workspace")
            
            if not os.path.isdir(abs_path):
                raise ValueError(f"Not a directory: {directory_path}")
            
            items = os.listdir(abs_path)
            return items
        except Exception as e:
            raise ValueError(f"Error listing directory: {str(e)}")
    
    def _search_web(self, query: str, max_results: int = 5) -> str:
        """Search the web (mock implementation)"""
        # This is a mock implementation
        # In production, you would call a real search API
        return f"Search results for '{query}': (mock - implement with real search API)"


# Global tool executor instance
tool_executor = ToolExecutor()

