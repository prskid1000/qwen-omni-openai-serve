"""
Gradio Interface for Qwen2.5-Omni Server
Multimodal chat interface supporting text, audio, image, and video inputs
"""

import gradio as gr
import requests
import json
import os
from typing import Optional

# Server configuration
SERVER_URL = os.getenv("OMNI_SERVER_URL", "http://localhost:8665")


def check_server_status():
    """Check if the Omni server is running"""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data
        return False, None
    except:
        return False, None


def omni_chat(
    text: str,
    audio_file,
    image_file,
    video_file,
    max_tokens: int,
    temperature: float,
    top_p: float,
    return_audio: bool
):
    """Send multimodal chat request to Omni server"""
    
    # Check server status first
    is_healthy, health_data = check_server_status()
    if not is_healthy:
        return "❌ Cannot connect to Omni server. Make sure it's running at " + SERVER_URL + "\n\nStart it with: python -m app.main", None
    
    # Validate inputs
    if not text or not text.strip():
        return "⚠️ Please enter a text prompt", None
    
    try:
        # Prepare form data
        files = {}
        data = {
            "text": text.strip(),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "response_format_type": "audio" if return_audio else "text"  # OpenAI-compatible format
        }
        
        # Add file uploads if provided
        if audio_file:
            files["audio"] = open(audio_file, "rb")
        
        if image_file:
            files["image"] = open(image_file, "rb")
        
        if video_file:
            files["video"] = open(video_file, "rb")
        
        # Send request
        response = requests.post(
            f"{SERVER_URL}/v1/omni/chat/completions/upload",
            files=files,
            data=data,
            timeout=300
        )
        
        # Close file handles
        for file_handle in files.values():
            file_handle.close()
        
        if response.status_code != 200:
            error_msg = response.text
            try:
                error_json = response.json()
                error_msg = error_json.get("detail", error_msg)
            except:
                pass
            return f"❌ Error {response.status_code}: {error_msg}", None
        
        # Parse response
        result = response.json()
        
        # Extract text response and audio (OpenAI format)
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            text_response = message.get("content", "")
            
            # Extract audio from OpenAI format: message.audio.data
            audio_data_url = None
            if "audio" in message:
                audio_data = message["audio"].get("data")
                if audio_data:
                    audio_data_url = f"data:audio/wav;base64,{audio_data}"
            
            # Add usage info if available
            usage = result.get("usage", {})
            if usage:
                usage_info = f"\n\n---\n📊 Usage: {usage.get('prompt_tokens', 0)} prompt tokens, {usage.get('completion_tokens', 0)} completion tokens, {usage.get('total_tokens', 0)} total"
                text_response += usage_info
            
            # Return tuple: (text, audio_data_url)
            return text_response, audio_data_url
        else:
            return "⚠️ No response generated", None
            
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. The model may be processing a large input. Please try again.", None
    except requests.exceptions.ConnectionError:
        return f"❌ Connection Error: Cannot connect to server at {SERVER_URL}\n\nMake sure the Omni server is running:\npython -m app.main", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None


def create_interface():
    """Create the Gradio interface for Omni multimodal chat"""
    
    # Check server status on load
    is_healthy, health_data = check_server_status()
    status_message = ""
    if is_healthy:
        model_name = health_data.get("model_name", "Unknown")
        device = health_data.get("device", "Unknown")
        context_length = health_data.get("context_length")
        context_info = f"\n📏 **Context Length**: {context_length:,} tokens" if context_length else ""
        status_message = f"✅ **Server Status**: Connected\n📚 **Model**: {model_name}\n🖥️ **Device**: {device}{context_info}"
    else:
        status_message = f"⚠️ **Server Status**: Not connected\n\nPlease start the server:\n```bash\npython -m app.main\n```"
    
    with gr.Blocks(title="Qwen2.5-Omni Multimodal Chat", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎯 Qwen2.5-Omni Multimodal Chat")
        gr.Markdown("Chat with Qwen2.5-Omni using text, audio, images, and video inputs. **OpenAI-compatible API** with audio output support.")
        
        # Server status
        with gr.Row():
            status_display = gr.Markdown(status_message)
            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")
        
        def refresh_status():
            is_healthy, health_data = check_server_status()
            if is_healthy:
                model_name = health_data.get("model_name", "Unknown")
                device = health_data.get("device", "Unknown")
                context_length = health_data.get("context_length")
                context_info = f"\n📏 **Context Length**: {context_length:,} tokens" if context_length else ""
                return f"✅ **Server Status**: Connected\n📚 **Model**: {model_name}\n🖥️ **Device**: {device}{context_info}"
            else:
                return f"⚠️ **Server Status**: Not connected\n\nPlease start the server:\n```bash\npython -m app.main\n```"
        
        refresh_btn.click(refresh_status, outputs=status_display)
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Inputs")
                
                # Text input (required)
                text_input = gr.Textbox(
                    label="Text Prompt *",
                    placeholder="Enter your question or instruction here...",
                    lines=5,
                    value=""
                )
                
                # Audio input
                audio_input = gr.Audio(
                    label="Audio Input (optional)",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                # Image input
                image_input = gr.Image(
                    label="Image Input (optional)",
                    type="filepath",
                    sources=["upload"]
                )
                
                # Video input
                video_input = gr.Video(
                    label="Video Input (optional)",
                    sources=["upload"]
                )
                
                gr.Markdown("### ⚙️ Parameters")
                
                with gr.Row():
                    max_tokens = gr.Slider(
                        minimum=1,
                        maximum=4096,
                        value=512,
                        step=1,
                        label="Max Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p"
                    )
                
                # Audio output toggle (OpenAI-compatible: response_format)
                return_audio = gr.Checkbox(
                    label="🎵 Generate Audio Output",
                    value=False,
                    info="Uses OpenAI-compatible response_format={'type': 'audio'}. Model will reload when switching modes."
                )
                
                send_btn = gr.Button("🚀 Send", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")
                
                output_text = gr.Textbox(
                    label="Text Response",
                    lines=15,
                    interactive=False,
                    placeholder="Response will appear here..."
                )
                
                # Audio output player
                output_audio = gr.Audio(
                    label="Audio Response (if generated)",
                    type="filepath",
                    visible=True,
                    autoplay=True
                )
        
        # Button actions
        def handle_chat_response(text, audio_file, image_file, video_file, max_tokens, temperature, top_p, return_audio):
            """Handle chat response and extract audio"""
            text_result, audio_url = omni_chat(
                text, audio_file, image_file, video_file, max_tokens, temperature, top_p, return_audio
            )
            
            # Save audio to temp file if present
            audio_file_path = None
            if audio_url:
                try:
                    import base64
                    import tempfile
                    import os
                    
                    # Extract base64 from data URL
                    if audio_url.startswith("data:audio/wav;base64,"):
                        audio_base64 = audio_url.split(",", 1)[1]
                        audio_bytes = base64.b64decode(audio_base64)
                        
                        # Save to temp file
                        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        temp_audio.write(audio_bytes)
                        temp_audio.close()
                        audio_file_path = temp_audio.name
                except Exception as e:
                    print(f"⚠️ Failed to save audio: {e}")
            
            return text_result, audio_file_path
        
        send_btn.click(
            fn=handle_chat_response,
            inputs=[text_input, audio_input, image_input, video_input, max_tokens, temperature, top_p, return_audio],
            outputs=[output_text, output_audio]
        )
        
        clear_btn.click(
            fn=lambda: ("", None, None, None, 512, 0.7, 0.9, False, "", None),
            outputs=[text_input, audio_input, image_input, video_input, max_tokens, temperature, top_p, return_audio, output_text, output_audio]
        )
        
        gr.Markdown("---")
        gr.Markdown("""
        ### 📝 Notes
        
        - **Text input is required** - All requests must include a text prompt
        - **Multimodal inputs are optional** - You can combine text with audio, image, and/or video
        - **Supported formats**:
          - Audio: WAV, MP3, FLAC, etc.
          - Image: PNG, JPG, JPEG, etc.
          - Video: MP4, AVI, MOV, etc.
        - **Output**: Text and optional audio (enable "Generate Audio Output" checkbox)
          - When audio is enabled, response includes both `message.content` (text) and `message.audio.data` (base64 audio)
          - Uses OpenAI-compatible format: `response_format={"type": "audio"}`
        - **Audio Mode**: Model will automatically reload when switching between text-only and audio modes
        - **API**: OpenAI-compatible - Use `response_format={"type": "audio"}` to request audio output
        - **Server**: Make sure the Omni server is running on port 8665
        """)
    
    return app


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
