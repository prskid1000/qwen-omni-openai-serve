"""
Omni Chat API Routes
POST /v1/omni/chat/completions - Multimodal chat with Qwen2.5-Omni
"""

import time
import uuid
import base64
import io
from typing import TYPE_CHECKING, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path

from ..models import OmniChatRequest, OmniChatResponse

if TYPE_CHECKING:
    from ..omni_manager import OmniModelManager

router = APIRouter()
omni_manager: 'OmniModelManager' = None


def set_omni_manager(manager):
    """Set the Omni model manager instance"""
    global omni_manager
    omni_manager = manager


@router.post("/v1/omni/chat/completions")
async def omni_chat_completions(request: OmniChatRequest) -> OmniChatResponse:
    """Create chat completion with Qwen2.5-Omni (multimodal support)"""
    
    if not omni_manager:
        raise HTTPException(status_code=500, detail="Omni model not loaded")
    
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required")
    
    # Get the last user message (for now, we'll process the last message)
    last_message = request.messages[-1]
    
    if last_message.role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")
    
    print(f"📨 Omni Chat: text_len={len(last_message.content)}, "
          f"audio={bool(last_message.audio_path)}, "
          f"image={bool(last_message.image_path)}, "
          f"video={bool(last_message.video_path)}, "
          f"return_audio={request.return_audio}")
    
    try:
        # Generate response
        response_text, audio_tensor = omni_manager.generate_response(
            text_prompt=last_message.content,
            audio_path=last_message.audio_path,
            image_path=last_message.image_path,
            video_path=last_message.video_path,
            max_new_tokens=request.max_tokens,
            return_audio=request.return_audio,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Encode audio if present
        audio_base64 = None
        if audio_tensor is not None:
            try:
                import soundfile as sf
                import numpy as np
                
                # Convert tensor to numpy
                audio_np = audio_tensor.reshape(-1).detach().cpu().numpy()
                
                # Write to bytes buffer
                audio_buffer = io.BytesIO()
                sf.write(audio_buffer, audio_np.reshape(-1, 1), 24000, format='WAV', subtype='PCM_16')
                audio_buffer.seek(0)
                audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
                print(f"✅ Audio generated: {len(audio_base64)} bytes (base64)")
            except Exception as e:
                print(f"⚠️  Failed to encode audio: {e}")
        
        # Estimate tokens (simple word count)
        prompt_tokens = len(last_message.content.split())
        completion_tokens = len(response_text.split())
        
        return OmniChatResponse(
            id=f"omni-{uuid.uuid4().hex[:8]}",
            model=omni_manager.model_name,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            audio_base64=audio_base64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/v1/omni/chat/completions/upload")
async def omni_chat_with_upload(
    text: str = Form(...),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    max_tokens: int = Form(512),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    return_audio: bool = Form(False)
) -> OmniChatResponse:
    """Create chat completion with file uploads (multimodal)"""
    
    if not omni_manager:
        raise HTTPException(status_code=500, detail="Omni model not loaded")
    
    print(f"📨 Omni Chat (Upload): text_len={len(text)}, "
          f"audio={audio is not None}, "
          f"image={image is not None}, "
          f"video={video is not None}, "
          f"return_audio={return_audio}")
    
    # Save uploaded files temporarily
    temp_files = []
    audio_path = None
    image_path = None
    video_path = None
    
    try:
        if audio:
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename).suffix)
            content = await audio.read()
            temp_audio.write(content)
            temp_audio.close()
            audio_path = temp_audio.name
            temp_files.append(audio_path)
        
        if image:
            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix)
            content = await image.read()
            temp_image.write(content)
            temp_image.close()
            image_path = temp_image.name
            temp_files.append(image_path)
        
        if video:
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix)
            content = await video.read()
            temp_video.write(content)
            temp_video.close()
            video_path = temp_video.name
            temp_files.append(video_path)
        
        # Generate response
        response_text, audio_tensor = omni_manager.generate_response(
            text_prompt=text,
            audio_path=audio_path,
            image_path=image_path,
            video_path=video_path,
            max_new_tokens=max_tokens,
            return_audio=return_audio,
            temperature=temperature,
            top_p=top_p
        )
        
        # Encode audio if present
        audio_base64 = None
        if audio_tensor is not None:
            try:
                import soundfile as sf
                import numpy as np
                
                audio_np = audio_tensor.reshape(-1).detach().cpu().numpy()
                audio_buffer = io.BytesIO()
                sf.write(audio_buffer, audio_np.reshape(-1, 1), 24000, format='WAV', subtype='PCM_16')
                audio_buffer.seek(0)
                audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
                print(f"✅ Audio generated: {len(audio_base64)} bytes (base64)")
            except Exception as e:
                print(f"⚠️  Failed to encode audio: {e}")
        
        # Estimate tokens
        prompt_tokens = len(text.split())
        completion_tokens = len(response_text.split())
        
        return OmniChatResponse(
            id=f"omni-{uuid.uuid4().hex[:8]}",
            model=omni_manager.model_name,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            audio_base64=audio_base64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    finally:
        # Cleanup temp files
        import os
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass

