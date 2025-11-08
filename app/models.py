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


class OmniChatRequest(BaseModel):
    """Request for Omni chat completion"""
    messages: List[OmniChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    return_audio: bool = Field(default=False, description="Generate audio output (requires talker to be enabled)")


class OmniChatResponse(BaseModel):
    """Response from Omni chat completion"""
    id: str
    model: str
    choices: List[Dict[str, Any]]  # Contains text response
    usage: Dict[str, int]
    audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio (only if return_audio=True and talker enabled)")


class OmniHealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    device: Optional[str] = None
    context_length: Optional[int] = None

