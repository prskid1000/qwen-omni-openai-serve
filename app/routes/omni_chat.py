"""
Omni Chat API Routes
POST /v1/omni/chat/completions - Multimodal chat with Qwen2.5-Omni
"""

import time
import uuid
import base64
import io
import json
import re
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path

from ..models import OmniChatRequest, OmniChatResponse, OmniChatMessage
from ..tool_service import tool_service

if TYPE_CHECKING:
    from ..omni_manager import OmniModelManager

router = APIRouter()
omni_manager: 'OmniModelManager' = None


def set_omni_manager(manager):
    """Set the Omni model manager instance"""
    global omni_manager
    omni_manager = manager


async def convert_base64_to_temp_file(base64_data: str, suffix: str = ".tmp") -> Optional[str]:
    """Convert base64 data (or data URL) to a temporary file and return the path"""
    if not base64_data:
        return None
    
    try:
        # Handle data URL format (data:image/png;base64,...)
        if base64_data.startswith("data:"):
            # Extract the base64 part after the comma
            base64_part = base64_data.split(",", 1)[1]
        else:
            base64_part = base64_data
        
        # Decode base64
        file_bytes = base64.b64decode(base64_part)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(file_bytes)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        print(f"⚠️  Failed to convert base64 to temp file: {e}")
        return None


@router.get("/v1/omni/tools")
async def get_available_tools():
    """Get list of available tools (includes built-in and MCP tools)"""
    tools = await tool_service.get_available_tools()
    return {
        "tools": tools
    }


def parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from model text response (JSON format)"""
    tool_calls = []
    
    if not text:
        return tool_calls
    
    print(f"🔍 Parsing tool calls from text (length: {len(text)})")
    print(f"📝 Text preview: {text[:200]}...")
    
    # Look for JSON tool call patterns like: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    # More flexible pattern that handles whitespace
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    print(f"🔍 Found {len(matches)} potential tool call matches with <tool_call> tags")
    
    for match in matches:
        try:
            cleaned_match = match.strip()
            print(f"🔍 Attempting to parse: {cleaned_match[:100]}...")
            tool_data = json.loads(cleaned_match)
            if "name" in tool_data and "arguments" in tool_data:
                tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_calls.append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_data["name"],
                        "arguments": json.dumps(tool_data["arguments"]) if isinstance(tool_data["arguments"], dict) else str(tool_data["arguments"])
                    }
                })
                print(f"✅ Successfully parsed tool call: {tool_data['name']}")
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON decode error: {e}")
            continue
        except Exception as e:
            print(f"⚠️  Error parsing tool call: {e}")
            continue
    
    # Also try to parse standalone JSON objects that look like tool calls
    if not tool_calls:
        # More flexible pattern for JSON objects with name and arguments
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        print(f"🔍 Found {len(json_matches)} potential standalone JSON matches")
        
        for match in json_matches:
            try:
                tool_data = json.loads(match)
                if "name" in tool_data and "arguments" in tool_data:
                    tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                    tool_calls.append({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_data["name"],
                            "arguments": json.dumps(tool_data["arguments"]) if isinstance(tool_data["arguments"], dict) else str(tool_data["arguments"])
                        }
                    })
                    print(f"✅ Successfully parsed standalone tool call: {tool_data['name']}")
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error for standalone: {e}")
                continue
            except Exception as e:
                print(f"⚠️  Error parsing standalone tool call: {e}")
                continue
    
    print(f"🔍 Final result: {len(tool_calls)} tool call(s) parsed")
    return tool_calls


async def execute_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Execute tool calls and return results"""
    tool_results = []
    
    for tool_call in tool_calls:
        tool_call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
        function = tool_call.get("function", {})
        tool_name = function.get("name", "")
        arguments_str = function.get("arguments", "{}")
        
        try:
            # Parse arguments
            if isinstance(arguments_str, str):
                arguments = json.loads(arguments_str)
            else:
                arguments = arguments_str
            
            # Execute tool (now async)
            result = await tool_service.execute_tool(tool_name, arguments)
            
            # Format result
            if isinstance(result, (dict, list)):
                result_str = json.dumps(result, ensure_ascii=False)
            else:
                result_str = str(result)
            
            tool_results.append({
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": tool_name,
                "content": result_str
            })
            
        except Exception as e:
            tool_results.append({
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": tool_name,
                "content": f"Error: {str(e)}"
            })
    
    return tool_results


@router.post("/v1/omni/chat/completions")
async def omni_chat_completions(request: OmniChatRequest) -> OmniChatResponse:
    """Create chat completion with Qwen2.5-Omni (multimodal support + tool calling)"""
    
    if not omni_manager:
        raise HTTPException(status_code=500, detail="Omni model not loaded")
    
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required")
    
    # Determine if audio is requested (OpenAI-compatible: response_format)
    wants_audio = False
    if request.response_format and request.response_format.type == "audio":
        wants_audio = True
    
    # Check if tools are provided
    has_tools = request.tools is not None and len(request.tools) > 0
    
    # Build conversation history for tool calling
    conversation_messages = request.messages.copy()
    max_iterations = 5  # Limit tool calling iterations
    iteration = 0
    
    # Track temporary files for cleanup
    temp_files_to_cleanup = []
    
    try:
        # Process all messages: convert base64 media to temp files for user messages
        # Ignore media outputs (audio_data) from assistant messages
        for msg in conversation_messages:
            if msg.role == "user":
                # Convert base64 media data to temp files for user messages
                if msg.audio_data and not msg.audio_path:
                    temp_path = await convert_base64_to_temp_file(msg.audio_data, suffix=".wav")
                    if temp_path:
                        msg.audio_path = temp_path
                        temp_files_to_cleanup.append(temp_path)
                
                if msg.image_data and not msg.image_path:
                    # Try to detect image format from data URL
                    suffix = ".png"
                    if msg.image_data.startswith("data:image/"):
                        mime_type = msg.image_data.split(";")[0].split("/")[1]
                        suffix = f".{mime_type}" if mime_type in ["png", "jpg", "jpeg", "gif", "webp"] else ".png"
                    temp_path = await convert_base64_to_temp_file(msg.image_data, suffix=suffix)
                    if temp_path:
                        msg.image_path = temp_path
                        temp_files_to_cleanup.append(temp_path)
                
                if msg.video_data and not msg.video_path:
                    # Try to detect video format from data URL
                    suffix = ".mp4"
                    if msg.video_data.startswith("data:video/"):
                        mime_type = msg.video_data.split(";")[0].split("/")[1]
                        suffix = f".{mime_type}" if mime_type in ["mp4", "webm", "ogg"] else ".mp4"
                    temp_path = await convert_base64_to_temp_file(msg.video_data, suffix=suffix)
                    if temp_path:
                        msg.video_path = temp_path
                        temp_files_to_cleanup.append(temp_path)
            elif msg.role == "assistant":
                # Ignore media outputs from assistant messages (audio_data, image_data, video_data)
                # These are outputs, not inputs, so we don't process them
                pass
        
        # Reload model if switching between audio/text modes
        omni_manager.reload_model_if_needed(wants_audio)
        
        # Get language preference (default to English)
        language = getattr(request, 'language', 'en') or 'en'
        language_instruction = ""
        if language == 'en':
            language_instruction = "Please respond in English only."
        elif language == 'zh':
            language_instruction = "请用中文回答。"
        else:
            language_instruction = f"Please respond in {language}."
        
        # If tools are provided, add tool descriptions to system prompt
        # Note: LLM only sees tool names and descriptions, not MCP server details
        tool_prompt = ""
        if has_tools:
            # Clean tool schemas - remove any internal metadata
            clean_tool_schemas = []
            for tool in request.tools:
                tool_dict = tool.dict()
                # Remove any internal metadata (keys starting with _)
                clean_tool = {k: v for k, v in tool_dict.items() if not k.startswith("_")}
                if "function" in clean_tool:
                    clean_function = {k: v for k, v in clean_tool["function"].items() if not k.startswith("_")}
                    clean_tool["function"] = clean_function
                clean_tool_schemas.append(clean_tool)
            
            tool_prompt = f"\n\n{language_instruction}\n\nAvailable tools:\n"
            for tool in request.tools:
                tool_prompt += f"- {tool.function.name}: {tool.function.description}\n"
            tool_prompt += "\nTo use a tool, format your response as:\n<tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>\n"
        else:
            # Add language instruction even without tools
            if language_instruction:
                tool_prompt = f"\n\n{language_instruction}\n"
        
        # Process messages and handle tool calls iteratively
        final_response = None
        final_audio = None
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get the last message for processing
            last_message = conversation_messages[-1]
            
            # Build proper conversation array with all messages and their media
            # This is the format expected by the model's apply_chat_template
            conversation_array = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Please respond in English unless the user explicitly asks for another language."}
                    ]
                }
            ]
            
            # Add language instruction and tool information to system message on first iteration
            if iteration == 1:
                if tool_prompt:
                    conversation_array[0]["content"][0]["text"] += tool_prompt
                elif language_instruction:
                    conversation_array[0]["content"][0]["text"] += f"\n\n{language_instruction}"
            
            # Process all messages and build conversation array
            for msg in conversation_messages:
                if msg.role == "user":
                    # Build user message with all media inputs
                    user_content = []
                    
                    # Add media inputs (ignore media outputs)
                    if msg.audio_path:
                        user_content.append({
                            "type": "audio",
                            "audio": msg.audio_path
                        })
                    if msg.image_path:
                        user_content.append({
                            "type": "image",
                            "image": msg.image_path
                        })
                    if msg.video_path:
                        user_content.append({
                            "type": "video",
                            "video": msg.video_path
                        })
                    
                    # Add text content (even if empty, to maintain conversation structure)
                    user_content.append({
                        "type": "text",
                        "text": msg.content or ""
                    })
                    
                    # Always add user message to maintain conversation flow
                    conversation_array.append({
                        "role": "user",
                        "content": user_content
                    })
                
                elif msg.role == "assistant":
                    # Add assistant message (text only, no media outputs)
                    if msg.content:
                        conversation_array.append({
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": msg.content}
                            ]
                        })
                
                elif msg.role == "tool":
                    # Tool results are included in the conversation context as text
                    # Format: "Tool Result (tool_call_id): content"
                    if msg.content:
                        conversation_array.append({
                            "role": "user",  # Tool results are treated as user input
                            "content": [
                                {"type": "text", "text": f"Tool Result ({msg.tool_call_id}): {msg.content}"}
                            ]
                        })
            
            # Generate response using the full conversation array
            response_text, audio_tensor = omni_manager.generate_response(
                conversation=conversation_array,
                max_new_tokens=request.max_tokens,
                return_audio=wants_audio,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            # Store final response (will be overwritten if we continue)
            final_response = response_text
            final_audio = audio_tensor
            
            # Check for tool calls in response
            tool_calls = parse_tool_calls_from_text(response_text)
            
            print(f"🔍 Iteration {iteration}: Found {len(tool_calls) if tool_calls else 0} tool call(s)")
            
                # If no tool calls, return the final response
            if not tool_calls or not has_tools:
                # Clean up tool call markers from response
                cleaned_text = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL).strip()
                if not cleaned_text:
                    cleaned_text = final_response
                
                # Encode audio if present
                audio_base64 = None
                if final_audio is not None:
                    try:
                        import soundfile as sf
                        import numpy as np
                        
                        audio_np = final_audio.reshape(-1).detach().cpu().numpy()
                        audio_buffer = io.BytesIO()
                        sf.write(audio_buffer, audio_np.reshape(-1, 1), 24000, format='WAV', subtype='PCM_16')
                        audio_buffer.seek(0)
                        audio_bytes = audio_buffer.read()
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        
                        print(f"✅ Audio generated: {len(audio_base64)} bytes (base64)")
                    except Exception as e:
                        print(f"⚠️  Failed to encode audio: {e}")
                
                # Build conversation messages for UI - only include tool-related messages if present
                # For normal responses, we don't need to send all history back to UI
                # The UI already has the history, we just need to send the NEW response
                conversation_for_ui = None  # Only set if we have tool calls/results
                
                # Only include conversation_messages if there were tool calls (for debugging/tool flow)
                if tool_calls or any(msg.role == "tool" for msg in conversation_messages):
                    conversation_for_ui = []
                    # Only include tool-related messages, not all history
                    for msg in conversation_messages:
                        if msg.role in ["tool"] or (msg.role == "assistant" and msg.tool_calls):
                            msg_dict = {
                                "role": msg.role,
                                "content": msg.content,
                            }
                            if msg.tool_calls:
                                msg_dict["tool_calls"] = msg.tool_calls
                            if msg.tool_call_id:
                                msg_dict["tool_call_id"] = msg.tool_call_id
                            conversation_for_ui.append(msg_dict)
                    
                    # Add the final assistant response
                    final_msg_dict = {
                        "role": "assistant",
                        "content": cleaned_text
                    }
                    if tool_calls:
                        final_msg_dict["tool_calls"] = tool_calls
                    conversation_for_ui.append(final_msg_dict)
                
                # Estimate tokens
                prompt_tokens = sum(len(msg.content.split()) if msg.content else 0 for msg in conversation_messages)
                completion_tokens = len(cleaned_text.split())
                
                # Build message according to OpenAI format
                message = {
                    "role": "assistant",
                    "content": cleaned_text
                }
                
                # Add tool calls if present (from first iteration)
                if tool_calls:
                    message["tool_calls"] = tool_calls
                
                # If audio is present, add it in OpenAI format
                if audio_base64:
                    message["audio"] = {
                        "data": audio_base64,
                        "format": "wav"
                    }
                
                finish_reason = "tool_calls" if tool_calls else "stop"
                
                return OmniChatResponse(
                    id=f"omni-{uuid.uuid4().hex[:8]}",
                    model=omni_manager.model_name,
                    choices=[{
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason
                    }],
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    },
                    conversation_messages=conversation_for_ui,
                    audio_base64=audio_base64  # Keep for backward compatibility
                )
            
            # Execute tool calls
            print(f"🔧 Executing {len(tool_calls)} tool call(s)...")
            tool_results = await execute_tool_calls(tool_calls)
            
            # Add assistant message with tool calls to conversation
            assistant_msg = OmniChatMessage(
                role="assistant",
                content=response_text,
                tool_calls=tool_calls
            )
            conversation_messages.append(assistant_msg)
            
            # Add tool results to conversation
            tool_results_text = "Tool results:\n"
            for tool_result in tool_results:
                tool_msg = OmniChatMessage(
                    role="tool",
                    content=tool_result["content"],
                    tool_call_id=tool_result["tool_call_id"]
                )
                conversation_messages.append(tool_msg)
                tool_results_text += f"- {tool_result['name']}: {tool_result['content']}\n"
            
            # Create a new user message with tool results to continue the conversation
            continue_msg = OmniChatMessage(
                role="user",
                content=f"Based on the tool results, please provide a final answer:\n{tool_results_text}"
            )
            conversation_messages.append(continue_msg)
            
            # Continue generation with tool results (next iteration)
            # The loop will continue and generate a new response with the tool results
            continue
        
        # If we've exhausted iterations, return the last response we got
        if final_response:
            cleaned_text = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL).strip()
            if not cleaned_text:
                cleaned_text = final_response
            
            # Build conversation messages for UI - only include tool-related messages if present
            conversation_for_ui = None  # Only set if we have tool calls/results
            
            # Encode audio if present
            audio_base64 = None
            if final_audio is not None:
                try:
                    import soundfile as sf
                    import numpy as np
                    
                    audio_np = final_audio.reshape(-1).detach().cpu().numpy()
                    audio_buffer = io.BytesIO()
                    sf.write(audio_buffer, audio_np.reshape(-1, 1), 24000, format='WAV', subtype='PCM_16')
                    audio_buffer.seek(0)
                    audio_bytes = audio_buffer.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                except Exception as e:
                    print(f"⚠️  Failed to encode audio: {e}")
            
            prompt_tokens = sum(len(msg.content.split()) if msg.content else 0 for msg in conversation_messages)
            completion_tokens = len(cleaned_text.split())
            
            return OmniChatResponse(
                id=f"omni-{uuid.uuid4().hex[:8]}",
                model=omni_manager.model_name,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": cleaned_text
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                conversation_messages=conversation_for_ui,
                audio_base64=audio_base64
            )
        
        raise HTTPException(status_code=500, detail="Maximum tool calling iterations reached")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    finally:
        # Cleanup temporary files created from base64 data
        import os
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"⚠️  Failed to cleanup temp file {temp_file}: {e}")


@router.post("/v1/omni/chat/completions/upload")
async def omni_chat_with_upload(
    text: str = Form(...),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    max_tokens: int = Form(512),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    response_format_type: Optional[str] = Form(None, description="Response format type: 'text' or 'audio'")
) -> OmniChatResponse:
    """Create chat completion with file uploads (multimodal)"""
    
    if not omni_manager:
        raise HTTPException(status_code=500, detail="Omni model not loaded")
    
    # Determine if audio is requested (OpenAI-compatible: response_format_type)
    wants_audio = response_format_type == "audio"
    
    print(f"📨 Omni Chat (Upload): text_len={len(text)}, "
          f"audio={audio is not None}, "
          f"image={image is not None}, "
          f"video={video is not None}, "
          f"response_format={response_format_type or 'text'}, "
          f"wants_audio={wants_audio}")
    
    # Save uploaded files temporarily
    # Reload model if switching between audio/text modes
    omni_manager.reload_model_if_needed(wants_audio)
    
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
            return_audio=wants_audio,
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
                audio_bytes = audio_buffer.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                print(f"✅ Audio generated: {len(audio_base64)} bytes (base64)")
            except Exception as e:
                print(f"⚠️  Failed to encode audio: {e}")
        
        # Estimate tokens
        prompt_tokens = len(text.split())
        completion_tokens = len(response_text.split())
        
        # Build message according to OpenAI format
        message = {
            "role": "assistant",
            "content": response_text
        }
        
        # If audio is present, add it in OpenAI format: message.audio.data
        if audio_base64:
            message["audio"] = {
                "data": audio_base64,
                "format": "wav"
            }
        
        return OmniChatResponse(
            id=f"omni-{uuid.uuid4().hex[:8]}",
            model=omni_manager.model_name,
            choices=[{
                "index": 0,
                "message": message,
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
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

