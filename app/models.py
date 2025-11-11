"""
Pydantic Models for Omni API
Request/response schemas
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


class OmniChatMessage(BaseModel):
    """Chat message with multimodal support"""
    role: str  # "user", "assistant", "system", or "tool"
    content: Optional[str] = None
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    # Base64 media data (alternative to paths, for UI history)
    audio_data: Optional[str] = None  # base64 encoded audio data
    image_data: Optional[str] = None  # base64 encoded image data (data URL format)
    video_data: Optional[str] = None  # base64 encoded video data (data URL format)
    # Tool calling support (OpenAI-compatible)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None  # For tool result messages


class ResponseFormat(BaseModel):
    """OpenAI-compatible response format"""
    type: str = Field(default="text", description="Response format type: 'text' or 'audio'")


class ToolFunction(BaseModel):
    """Function definition for tool calling"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema


class Tool(BaseModel):
    """Tool definition (OpenAI-compatible)"""
    type: str = "function"
    function: ToolFunction


class OmniChatRequest(BaseModel):
    """Request for Omni chat completion (OpenAI-compatible)"""
    messages: List[OmniChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    response_format: Optional[ResponseFormat] = Field(default=None, description="OpenAI-compatible response format: {'type': 'text'} or {'type': 'audio'}")
    tools: Optional[List[Tool]] = Field(default=None, description="List of available tools for function calling")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Tool choice: 'none', 'auto', or specific tool")
    language: Optional[str] = Field(default="en", description="Response language code (e.g., 'en' for English, 'zh' for Chinese). Default: 'en'")


class OmniChatResponse(BaseModel):
    """Response from Omni chat completion (OpenAI-compatible format)"""
    id: str
    model: str
    choices: List[Dict[str, Any]]  # Contains text/audio response in OpenAI format (message.audio.data for audio)
    usage: Dict[str, int]
    conversation_messages: Optional[List[Dict[str, Any]]] = Field(default=None, description="Full conversation including tool calls and results")
    
    class Config:
        # Allow extra fields for OpenAI compatibility
        extra = "allow"


class OmniHealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    device: Optional[str] = None
    context_length: Optional[int] = None


class ToolExecutionResult(BaseModel):
    """Result of tool execution"""
    tool_call_id: str
    name: str
    result: Any
    error: Optional[str] = None


# MCP Server Management Models
class MCPServerConfig(BaseModel):
    """MCP Server configuration"""
    # STDIO config
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    
    # HTTP config
    url: Optional[str] = None
    prefer_sse: Optional[bool] = False
    
    # Common config
    timeout: Optional[int] = None
    version: Optional[str] = None


class MCPServerConnectRequest(BaseModel):
    """Request to connect to an MCP server"""
    server_id: str
    server_config: MCPServerConfig


class MCPServerSummary(BaseModel):
    """Summary of an MCP server"""
    id: str
    status: str  # "disconnected", "connecting", "connected"
    config: Dict[str, Any]
    error: Optional[str] = None


class MCPServerListResponse(BaseModel):
    """Response listing all MCP servers"""
    servers: List[MCPServerSummary]


class MCPServerConnectResponse(BaseModel):
    """Response from connecting to an MCP server"""
    success: bool
    status: str
    error: Optional[str] = None
