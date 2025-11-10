"""
Tool Management Service
Handles tool registration, discovery, and execution
"""

from typing import Dict, Any, List, Optional, Callable
from .tool_executor import ToolExecutor, tool_executor


class ToolService:
    """Service for managing tools"""
    
    def __init__(self, executor: Optional[ToolExecutor] = None):
        """
        Initialize tool service
        
        Args:
            executor: Optional ToolExecutor instance. If None, uses global instance.
        """
        self.executor = executor or tool_executor
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of all available tools with their schemas
        
        Returns:
            List of tool schemas in OpenAI-compatible format
        """
        return self.executor.get_tool_schemas()
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool with given arguments
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments for the tool
            
        Returns:
            Result of tool execution
            
        Raises:
            ValueError: If tool is not found
            Exception: If tool execution fails
        """
        return self.executor.execute_tool(tool_name, arguments)
    
    def register_tool(self, name: str, func: Callable, description: str = "", parameters: Optional[Dict[str, Any]] = None):
        """
        Register a new tool dynamically
        
        Args:
            name: Tool name
            func: Tool function to execute
            description: Tool description (optional, for schema)
            parameters: Tool parameters schema (optional)
        """
        self.executor.register_tool(name, func)
        # Note: If you want to update schemas dynamically, you'd need to extend ToolExecutor
    
    def is_tool_available(self, tool_name: str) -> bool:
        """
        Check if a tool is available
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if tool is available, False otherwise
        """
        return tool_name in self.executor.tool_registry
    
    def get_tool_count(self) -> int:
        """
        Get the number of available tools
        
        Returns:
            Number of registered tools
        """
        return len(self.executor.tool_registry)


# Global tool service instance
tool_service = ToolService()

