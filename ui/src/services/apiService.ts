import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8665';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  audioData?: string; // base64 audio data
  imageUrl?: string;
  videoUrl?: string;
  timestamp: number;
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
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
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
    }
  ): Promise<ChatResponse> {
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

  decodeAudioBase64(base64Data: string): string {
    // Convert base64 to data URL (persists across reloads)
    // HTML5 audio elements support data URLs
    return `data:audio/wav;base64,${base64Data}`;
  }
}

export const apiService = new ApiService();

