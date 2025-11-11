import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8665';

export interface ToolCall {
  id: string;
  type: string;
  function: {
    name: string;
    arguments: string;
  };
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'tool' | 'system';
  content: string;
  audioData?: string; // base64 audio data
  imageUrl?: string;
  videoUrl?: string;
  timestamp: number;
  toolCalls?: ToolCall[];
  toolCallId?: string;
}

export interface ChatResponse {
  id: string;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
      audio?: {
        data: string;
        format: string;
      };
      tool_calls?: ToolCall[];
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  conversation_messages?: Array<{
    role: string;
    content: string;
    tool_calls?: ToolCall[];
    tool_call_id?: string;
  }>;
}

export interface Tool {
  type: string;
  function: {
    name: string;
    description: string;
    parameters: Record<string, any>;
  };
}

export interface ToolsResponse {
  tools: Tool[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_name?: string;
  device?: string;
  context_length?: number;
}

class ApiService {
  private baseURL: string;

  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async checkHealth(): Promise<HealthResponse> {
    try {
      const response = await axios.get<HealthResponse>(`${this.baseURL}/health`);
      return response.data;
    } catch (error) {
      throw new Error('Failed to connect to server');
    }
  }

  async sendMessage(
    text: string,
    audioFile?: File,
    imageFile?: File,
    videoFile?: File,
    options?: {
      maxTokens?: number;
      temperature?: number;
      topP?: number;
      returnAudio?: boolean;
      tools?: Tool[];
      messages?: ChatMessage[];
    }
  ): Promise<ChatResponse> {
    // If tools are provided OR if messages (history) are provided, use the JSON API endpoint
    // This allows us to send chat history even when tool calling is disabled
    if ((options?.tools && options.tools.length > 0) || (options?.messages && options.messages.length > 0)) {
      return this.sendMessageWithTools(text, audioFile, imageFile, videoFile, options);
    }

    // Otherwise use the upload endpoint for backward compatibility
    const formData = new FormData();
    
    formData.append('text', text);
    
    if (audioFile) {
      formData.append('audio', audioFile);
    }
    
    if (imageFile) {
      formData.append('image', imageFile);
    }
    
    if (videoFile) {
      formData.append('video', videoFile);
    }
    
    formData.append('max_tokens', String(options?.maxTokens || 512));
    formData.append('temperature', String(options?.temperature || 0.7));
    formData.append('top_p', String(options?.topP || 0.9));
    
    if (options?.returnAudio) {
      formData.append('response_format_type', 'audio');
    }

    try {
      const response = await axios.post<ChatResponse>(
        `${this.baseURL}/v1/omni/chat/completions/upload`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: 300000, // 5 minutes timeout for long generations
        }
      );
      
      return response.data;
    } catch (error: any) {
      if (error.response) {
        throw new Error(error.response.data?.detail || 'Server error');
      } else if (error.request) {
        throw new Error('No response from server. Make sure the server is running.');
      } else {
        throw new Error(error.message || 'Request failed');
      }
    }
  }

  async sendMessageWithTools(
    text: string,
    audioFile?: File,
    imageFile?: File,
    videoFile?: File,
    options?: {
      maxTokens?: number;
      temperature?: number;
      topP?: number;
      returnAudio?: boolean;
      tools?: Tool[];
      messages?: ChatMessage[];
    }
  ): Promise<ChatResponse> {
    // Helper function to convert data URL to base64 string
    const dataUrlToBase64 = (dataUrl: string): string => {
      if (dataUrl.startsWith('data:')) {
        // Extract base64 part after comma
        return dataUrl.split(',')[1];
      }
      return dataUrl; // Already base64
    };

    // Helper function to convert File to base64
    const fileToBase64 = (file: File): Promise<string> => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result as string;
          // Extract base64 from data URL
          if (result.startsWith('data:')) {
            resolve(dataUrlToBase64(result));
          } else {
            resolve(result);
          }
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    };

    // Build messages array from history
    const messages: any[] = [];
    
    // Process history messages (excluding the current one)
    if (options?.messages && options.messages.length > 0) {
      for (const msg of options.messages) {
        const apiMsg: any = {
          role: msg.role,
          content: msg.content || '',
        };

        // For user messages: include media inputs (imageUrl, videoUrl, audioData)
        if (msg.role === 'user') {
          if (msg.imageUrl) {
            apiMsg.image_data = msg.imageUrl; // Send as data URL, backend will convert
          }
          if (msg.videoUrl) {
            apiMsg.video_data = msg.videoUrl; // Send as data URL, backend will convert
          }
          // Include audioData for user messages (voice input)
          if (msg.audioData) {
            // Convert audioData to proper format (might be base64 or data URL)
            if (msg.audioData.startsWith('data:')) {
              apiMsg.audio_data = msg.audioData;
            } else {
              // Assume it's base64, wrap in data URL
              apiMsg.audio_data = `data:audio/wav;base64,${msg.audioData}`;
            }
          }
        }
        
        // For assistant messages: exclude media outputs (audioData, imageUrl, videoUrl)
        // These are outputs, not inputs, so we don't send them
        // Only include text content and tool calls
        
        // Include tool calls and tool call IDs
        if (msg.toolCalls) {
          apiMsg.tool_calls = msg.toolCalls;
        }
        if (msg.toolCallId) {
          apiMsg.tool_call_id = msg.toolCallId;
        }

        messages.push(apiMsg);
      }
    }

    // Add the current message with media
    const currentMessage: any = {
      role: 'user',
      content: text || '',
    };

    // Convert current message's media files to base64
    if (audioFile) {
      const audioBase64 = await fileToBase64(audioFile);
      currentMessage.audio_data = `data:audio/wav;base64,${audioBase64}`;
    }
    if (imageFile) {
      const imageBase64 = await fileToBase64(imageFile);
      // Detect MIME type from file
      const mimeType = imageFile.type || 'image/png';
      currentMessage.image_data = `data:${mimeType};base64,${imageBase64}`;
    }
    if (videoFile) {
      const videoBase64 = await fileToBase64(videoFile);
      // Detect MIME type from file
      const mimeType = videoFile.type || 'video/mp4';
      currentMessage.video_data = `data:${mimeType};base64,${videoBase64}`;
    }

    messages.push(currentMessage);

    const requestBody = {
      messages,
      max_tokens: options?.maxTokens || 512,
      temperature: options?.temperature || 0.7,
      top_p: options?.topP || 0.9,
      tools: options?.tools,
      response_format: options?.returnAudio ? { type: 'audio' } : { type: 'text' },
    };

    try {
      const response = await axios.post<ChatResponse>(
        `${this.baseURL}/v1/omni/chat/completions`,
        requestBody,
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: 300000,
        }
      );
      
      return response.data;
    } catch (error: any) {
      if (error.response) {
        throw new Error(error.response.data?.detail || 'Server error');
      } else if (error.request) {
        throw new Error('No response from server. Make sure the server is running.');
      } else {
        throw new Error(error.message || 'Request failed');
      }
    }
  }

  async getAvailableTools(): Promise<ToolsResponse> {
    try {
      const response = await axios.get<ToolsResponse>(
        `${this.baseURL}/v1/omni/tools`
      );
      return response.data;
    } catch (error: any) {
      throw new Error('Failed to fetch available tools');
    }
  }

  decodeAudioBase64(base64Data: string): string {
    // Convert base64 to data URL (persists across reloads)
    // HTML5 audio elements support data URLs
    return `data:audio/wav;base64,${base64Data}`;
  }

  // MCP Server Management APIs
  async getMCPServers(): Promise<{ servers: Array<{ id: string; status: string; config: any; error?: string }> }> {
    try {
      const response = await axios.get(`${this.baseURL}/v1/mcp/servers`);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to fetch MCP servers');
    }
  }

  async connectMCPServer(serverId: string, serverConfig: any): Promise<{ success: boolean; status: string; error?: string }> {
    try {
      const response = await axios.post(`${this.baseURL}/v1/mcp/servers/connect`, {
        server_id: serverId,
        server_config: serverConfig,
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to connect MCP server');
    }
  }

  async disconnectMCPServer(serverId: string): Promise<void> {
    try {
      await axios.post(`${this.baseURL}/v1/mcp/servers/${serverId}/disconnect`);
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to disconnect MCP server');
    }
  }

  async removeMCPServer(serverId: string): Promise<void> {
    try {
      await axios.delete(`${this.baseURL}/v1/mcp/servers/${serverId}`);
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to remove MCP server');
    }
  }

  async getMCPServerStatus(serverId: string): Promise<{ server_id: string; status: string; config: any }> {
    try {
      const response = await axios.get(`${this.baseURL}/v1/mcp/servers/${serverId}/status`);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get MCP server status');
    }
  }
}

export const apiService = new ApiService();

