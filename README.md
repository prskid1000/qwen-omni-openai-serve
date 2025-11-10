# Omni Chat - Qwen2.5-Omni Multimodal AI Assistant

A powerful, full-stack AI chat application built with Qwen2.5-Omni, featuring multimodal input support, MCP (Model Context Protocol) server integration, and advanced tool calling capabilities.

## 🎯 Overview

Omni Chat is a modern AI assistant application that combines:
- **Qwen2.5-Omni Model**: A state-of-the-art multimodal AI model supporting text, audio, image, and video
- **FastAPI Backend**: High-performance Python server with OpenAI-compatible API
- **React Frontend**: Modern, responsive web interface with dark theme
- **MCP Integration**: Connect to external MCP servers for extended functionality
- **Tool Calling**: Built-in tools and MCP tools for enhanced capabilities

## ✨ Key Features

### 🎨 User Interface
- **Modern Dark Theme**: Sleek, professional design with Tailwind CSS
- **Chat History**: Persistent conversation history with sidebar navigation
- **Multimodal Input**: Support for text, audio, image, and video files
- **Voice Recording**: Record audio directly in the browser
- **Audio Playback**: Embedded audio player for text-to-speech responses
- **Responsive Design**: Works seamlessly on desktop and mobile devices

### 🤖 AI Capabilities
- **Multimodal Understanding**: Process text, audio, images, and videos in a single conversation
- **Tool Calling**: Automatic tool selection and execution based on user queries
- **Iterative Tool Use**: Multi-step reasoning with automatic tool chaining
- **Language Control**: Configurable response language (default: English)
- **Context Awareness**: Maintains conversation context across multiple turns

### 🔌 MCP Server Management
- **STDIO Transport**: Connect to local MCP servers via standard input/output
- **HTTP Transport**: Connect to remote MCP servers via HTTP/SSE
- **Automatic Tool Discovery**: Tools from connected MCP servers are automatically available
- **Connection Management**: Easy connect/disconnect/remove operations
- **Persistent Storage**: Server configurations saved in browser localStorage
- **Auto-Refresh**: Tool list automatically updates when servers connect/disconnect

### 🛠️ Built-in Tools
- `get_current_time`: Get the current date and time
- `calculate`: Perform mathematical calculations
- `get_weather`: Get weather information for a location
- `read_file`: Read contents of a file

### 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend (UI)                      │
│  - Chat Interface  - MCP Server Manager  - Tool Display     │
└───────────────────────────┬─────────────────────────────────┘
                             │ HTTP/REST API
┌───────────────────────────▼─────────────────────────────────┐
│                   FastAPI Backend (app/)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Omni Manager │  │ Tool Service │  │ MCP Manager  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘                │
│                          │                                   │
└──────────────────────────┼───────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
    │ Qwen2.5 │      │ Built-in │      │   MCP   │
    │  Omni   │      │  Tools   │      │ Servers │
    └─────────┘      └──────────┘      └─────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 18+** with npm
- **CUDA-capable GPU** (recommended) or CPU
- **8GB+ RAM** (16GB+ recommended)

### Backend Setup

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Optional: Install flash-attention** for faster inference:
```bash
pip install flash-attn --no-build-isolation
```

4. **Optional: Install qwen_omni_utils** for full multimodal support:
```bash
pip install qwen-omni-utils
```

5. **Start the backend server**:
```bash
python omni.py
```

The server will start on `http://localhost:8665` by default.

### Frontend Setup

1. **Navigate to the UI directory**:
```bash
cd ui
```

2. **Install dependencies**:
```bash
npm install
```

3. **Start the development server**:
```bash
npm run dev
```

The UI will be available at `http://localhost:3000` (or the next available port).

### Environment Variables

#### Backend (`omni.py` / `app/main.py`)
- `PORT`: Server port (default: `8665`)
- `HOST`: Server host (default: `0.0.0.0`)
- `RELOAD`: Enable auto-reload for development (default: `false`)
- `OMNI_MODEL_NAME`: Model name (default: `wolfofbackstreet/Qwen2.5-Omni-3B-4Bit`)
- `OMNI_USE_FLASH_ATTENTION`: Use flash attention (default: `true`)
- `OMNI_USE_CPU_OFFLOAD`: Use CPU offloading (default: `false`)

#### Frontend (`ui/.env`)
- `VITE_API_URL`: Backend API URL (default: `http://localhost:8665`)

## 📖 Usage Guide

### Starting a Conversation

1. Open the UI in your browser
2. Click **"+ New Chat"** to start a new conversation
3. Type your message in the input field
4. The AI will respond with text and optionally audio

### Using Multimodal Inputs

- **Text**: Simply type your message
- **Audio**: Click the microphone icon to record, or use the paperclip to upload an audio file
- **Image**: Use the paperclip icon to upload an image file
- **Video**: Use the paperclip icon to upload a video file

### Managing MCP Servers

1. **Add a Server**:
   - Click **"+ Add MCP Server"** in the MCP Servers section
   - Enter a Server ID (e.g., `search`)
   - Choose transport type (STDIO or HTTP)
   - For STDIO: Enter command (e.g., `npx`) and arguments
   - For HTTP: Enter the server URL
   - Click **"Add Server"**

2. **Connect a Server**:
   - Click the power icon next to a server to connect
   - Wait for the connection status to show "connected"
   - Tools from the server will automatically appear in the Available Tools section

3. **Disconnect/Remove**:
   - Click the power icon again to disconnect
   - Click the trash icon to remove a server configuration

### Using Tools

Tools are automatically available to the AI. When you ask a question that requires a tool, the AI will:
1. Identify which tool(s) to use
2. Execute the tool(s) automatically
3. Use the results to generate a response

**Example queries that trigger tools**:
- "What time is it?" → Uses `get_current_time`
- "Calculate 15 * 23 + 7" → Uses `calculate`
- "What's the weather in New York?" → Uses `get_weather`
- "Search the web for Python tutorials" → Uses MCP search tools (if connected)

## 🔌 MCP Server Integration

### What is MCP?

Model Context Protocol (MCP) is a protocol that enables AI assistants to securely access external data and tools. Omni Chat supports connecting to MCP servers to extend its capabilities.

### Supported Transports

#### STDIO Transport
For local MCP servers that run as processes:
```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-duckduckgo-search"],
  "env": {}
}
```

#### HTTP Transport
For remote MCP servers accessible via HTTP:
```json
{
  "url": "https://api.example.com/mcp",
  "prefer_sse": true
}
```

### Example MCP Servers

- **DuckDuckGo Search**: `npx -y @modelcontextprotocol/server-duckduckgo-search`
- **Filesystem**: `npx -y @modelcontextprotocol/server-filesystem`
- **GitHub**: `npx -y @modelcontextprotocol/server-github`

### MCP API Endpoints

- `GET /v1/mcp/servers` - List all MCP servers
- `POST /v1/mcp/servers/connect` - Connect to an MCP server
- `POST /v1/mcp/servers/{server_id}/disconnect` - Disconnect from a server
- `DELETE /v1/mcp/servers/{server_id}` - Remove a server
- `GET /v1/mcp/servers/{server_id}/status` - Get server status
- `GET /v1/mcp/servers/{server_id}/tools` - Get tools from a server
- `GET /v1/mcp/tools` - Get all tools from all servers

## 📡 API Documentation

### Chat Completions

**Endpoint**: `POST /v1/omni/chat/completions`

**Request**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?",
      "image_path": "/path/to/image.png",
      "audio_path": "/path/to/audio.wav",
      "video_path": "/path/to/video.mp4"
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "response_format": {
    "type": "text"
  },
  "tools": [...],
  "language": "en"
}
```

**Response**:
```json
{
  "id": "omni-abc123",
  "model": "Qwen2.5-Omni-3B",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "I'm doing well, thank you!",
      "audio": {
        "data": "base64_encoded_audio...",
        "format": "wav"
      }
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

### File Upload

**Endpoint**: `POST /v1/omni/chat/completions/upload`

**Form Data**:
- `text`: Text prompt (required)
- `audio`: Audio file (optional)
- `image`: Image file (optional)
- `video`: Video file (optional)
- `max_tokens`: Max tokens (default: 512)
- `temperature`: Temperature (default: 0.7)
- `top_p`: Top-p (default: 0.9)
- `response_format_type`: "text" or "audio" (default: "text")

### Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Qwen2.5-Omni-3B",
  "device": "cuda",
  "context_length": 32768
}
```

### Available Tools

**Endpoint**: `GET /v1/omni/tools`

**Response**:
```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_time",
        "description": "Get the current date and time",
        "parameters": {
          "type": "object",
          "properties": {},
          "required": []
        }
      }
    }
  ]
}
```

## 🛠️ Development

### Project Structure

```
omni/
├── app/                    # Backend FastAPI application
│   ├── main.py            # Main application entry point
│   ├── omni_manager.py    # Qwen2.5-Omni model manager
│   ├── mcp_client_manager.py  # MCP server client manager
│   ├── tool_service.py    # Tool management service
│   ├── tool_executor.py   # Built-in tool executor
│   ├── routes/            # API route handlers
│   │   ├── omni_chat.py   # Chat completion routes
│   │   └── mcp_servers.py # MCP server management routes
│   └── models.py          # Pydantic models
├── ui/                    # Frontend React application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API service layer
│   │   ├── hooks/         # Custom React hooks
│   │   └── utils/         # Utility functions
│   └── package.json       # Frontend dependencies
├── requirements.txt        # Python dependencies
├── omni.py                # Main startup script
└── README.md              # This file
```

### Running in Development Mode

**Backend**:
```bash
RELOAD=true python omni.py
```

**Frontend**:
```bash
cd ui
npm run dev
```

### Building for Production

**Frontend**:
```bash
cd ui
npm run build
```

The built files will be in `ui/dist/`.

## 🔧 Configuration

### Model Configuration

The default model is `wolfofbackstreet/Qwen2.5-Omni-3B-4Bit`, a 4-bit quantized version for lower memory usage. You can change this via the `OMNI_MODEL_NAME` environment variable.

### Tool Configuration

Built-in tools are automatically registered. MCP tools are discovered when servers connect. The tool list is automatically refreshed when:
- An MCP server connects
- An MCP server disconnects
- A manual refresh is triggered

### Language Configuration

By default, the model responds in English. You can change this by setting the `language` parameter in the chat request:
- `"en"` - English (default)
- `"zh"` - Chinese
- Other language codes as supported

## 📝 Notes

- **Talker Module**: Disabled by default to avoid meta tensor errors. Audio output requires the talker module to be enabled.
- **Flash Attention**: Enabled by default for faster inference but requires the `flash-attn` package.
- **Device Management**: The model uses `device_map="auto"` to automatically distribute across available GPUs/CPU.
- **Tool Privacy**: The LLM only sees tool names and descriptions, not internal MCP server details.
- **Connection Persistence**: MCP server configurations are saved in browser localStorage and persist across sessions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- **Qwen Team** at Alibaba Group for the Qwen2.5-Omni model
- **Model Context Protocol** for the MCP specification
- All open-source contributors and libraries used in this project

---

**Built with ❤️ using Qwen2.5-Omni, FastAPI, and React**
