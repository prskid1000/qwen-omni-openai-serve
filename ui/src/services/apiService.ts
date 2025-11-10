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
    // If tools are provided, use the JSON API endpoint
    if (options?.tools && options.tools.length > 0) {
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
    // Convert files to base64 if needed
    let audioPath: string | undefined;
    let imagePath: string | undefined;
    let videoPath: string | undefined;

    // For tool calling, we'll need to handle file uploads differently
    // For now, we'll use the JSON endpoint which expects file paths
    // In a real implementation, you'd upload files first and get paths

    const messages: any[] = options?.messages || [{
      role: 'user',
      content: text,
      audio_path: audioPath,
      image_path: imagePath,
      video_path: videoPath,
    }];

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
}

export const apiService = new ApiService();

