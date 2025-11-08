"""
Pydantic Models for Omni API
Request/response schemas
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class OmniChatMessage(BaseModel):
    """Chat message with multimodal support"""
    role: str  # "user" or "assistant"
    content: str
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    video_path: Optional[str] = None


class ResponseFormat(BaseModel):
    """OpenAI-compatible response format"""
    type: str = Field(default="text", description="Response format type: 'text' or 'audio'")

class OmniChatRequest(BaseModel):
    """Request for Omni chat completion (OpenAI-compatible)"""
    messages: List[OmniChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    response_format: Optional[ResponseFormat] = Field(default=None, description="OpenAI-compatible response format: {'type': 'text'} or {'type': 'audio'}")


class OmniChatResponse(BaseModel):
    """Response from Omni chat completion (OpenAI-compatible format)"""
    id: str
    model: str
    choices: List[Dict[str, Any]]  # Contains text/audio response in OpenAI format (message.audio.data for audio)
    usage: Dict[str, int]
    
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

