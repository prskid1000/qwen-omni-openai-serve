# App - Qwen2.5-Omni Model Server

FastAPI server for the Qwen2.5-Omni multimodal model with support for text, audio, image, and video inputs.

## Features

- **Multimodal Input Support**: Text, audio, image, and video
- **Audio Output**: Generate speech from text responses (when talker is enabled)
- **RESTful API**: OpenAI-compatible chat completions endpoint
- **File Upload**: Direct file upload support for multimodal inputs

## Installation

1. Install dependencies:
```bash
pip install fastapi uvicorn transformers torch soundfile pillow
```

2. Optional: Install flash-attention for faster inference:
```bash
pip install flash-attn --no-build-isolation
```

3. Optional: Install qwen_omni_utils for full multimodal support:
```bash
pip install qwen-omni-utils
```

## Usage

### Run the server:

**Option 1: Using the main startup script (recommended)**
```bash
python omni.py
```

**Option 2: Using the module**
```bash
python -m app.main
```

**Option 3: Using the app run script**
```bash
python app/run.py
```

### Environment Variables

- `PORT`: Server port (default: 8665)
- `HOST`: Server host (default: 0.0.0.0)
- `RELOAD`: Enable auto-reload for development (default: false)
- `OMNI_MODEL_NAME`: Model name (default: "Qwen/Qwen2.5-Omni-3B")
- `OMNI_USE_FLASH_ATTENTION`: Use flash attention (default: "true")
- `OMNI_USE_CPU_OFFLOAD`: Use CPU offloading (default: "false")

### Examples:

```bash
# Default port (8665)
python omni.py

# Custom port
PORT=9000 python omni.py

# Development mode with auto-reload
RELOAD=true python omni.py

# Custom host and port
HOST=127.0.0.1 PORT=8665 python omni.py
```

## API Endpoints

### POST `/v1/omni/chat/completions`

Chat completion with JSON request body.

**Request:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What do you see in this image?",
      "image_path": "/path/to/image.png"
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "return_audio": false
}
```

**Response:**
```json
{
  "id": "omni-abc123",
  "model": "Qwen/Qwen2.5-Omni-3B",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "I see..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  },
  "audio_base64": null
}
```

### POST `/v1/omni/chat/completions/upload`

Chat completion with file uploads (multipart/form-data).

**Form fields:**
- `text`: Text prompt (required)
- `audio`: Audio file (optional)
- `image`: Image file (optional)
- `video`: Video file (optional)
- `max_tokens`: Max tokens (default: 512)
- `temperature`: Temperature (default: 0.7)
- `top_p`: Top-p (default: 0.9)
- `return_audio`: Return audio (default: false)

### GET `/health`

Health check endpoint.

### GET `/`

Root endpoint with server info.

## Example Usage

### Text-only:
```python
import requests

response = requests.post("http://localhost:8665/v1/omni/chat/completions", json={
    "messages": [{
        "role": "user",
        "content": "Explain quantum computing"
    }]
})
print(response.json())
```

### With image upload:
```python
import requests

files = {
    "text": (None, "What do you see in this image?"),
    "image": open("image.png", "rb")
}
response = requests.post("http://localhost:8665/v1/omni/chat/completions/upload", files=files)
print(response.json())
```

### With audio output:
```python
import requests
import base64

response = requests.post("http://localhost:8665/v1/omni/chat/completions", json={
    "messages": [{
        "role": "user",
        "content": "Say hello in a friendly way"
    }],
    "return_audio": True
})

data = response.json()
if data.get("audio_base64"):
    audio_bytes = base64.b64decode(data["audio_base64"])
    with open("output.wav", "wb") as f:
        f.write(audio_bytes)
```

## Notes

- The talker module is disabled by default to avoid meta tensor errors. To enable audio output, you may need to modify the model loading code.
- Flash attention is enabled by default for faster inference but requires the `flash-attn` package.
- The model uses `device_map="auto"` to automatically distribute across available GPUs/CPU.

