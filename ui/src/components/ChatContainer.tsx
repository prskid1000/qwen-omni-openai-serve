import { useState, useEffect } from 'react';
import { Menu, X } from 'lucide-react';
import { ChatSidebar } from './ChatSidebar';
import { ChatArea } from './ChatArea';
import { MessageInput } from './MessageInput';
import { useChatHistory } from '../hooks/useChatHistory';
import { apiService } from '../services/apiService';

export function ChatContainer() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState<'online' | 'offline'>('online'); // Assume online, detect on error
  const [audioOutputEnabled, setAudioOutputEnabled] = useState(true); // Default to enabled
  
  const {
    chats,
    currentChatId,
    currentChat,
    createNewChat,
    addMessage,
  } = useChatHistory();

  // No automatic chat creation - user must click "New Chat" button
  // Server status will be determined when user tries to send a message

  const handleSend = async (
    text: string,
    audioFile?: File,
    imageFile?: File,
    videoFile?: File
  ) => {
    // Check if we have content to send
    if (!text.trim() && !imageFile && !videoFile && !audioFile) {
      return;
    }

    // Create a new chat if none exists
    let chatId = currentChatId;
    if (!chatId) {
      chatId = createNewChat();
    }

    setIsLoading(true);

    try {
      // Add user message to chat
      const userMessage = {
        role: 'user' as const,
        content: text || '[Media message]',
        timestamp: Date.now(),
        imageUrl: imageFile ? URL.createObjectURL(imageFile) : undefined,
        videoUrl: videoFile ? URL.createObjectURL(videoFile) : undefined,
      };

      addMessage(chatId, userMessage);

      // Update server status to online when sending
      if (serverStatus === 'offline') {
        setServerStatus('online');
      }

      // Send to API
      const response = await apiService.sendMessage(
        text || 'Describe the uploaded media.',
        audioFile,
        imageFile,
        videoFile,
        {
          returnAudio: audioOutputEnabled, // Use user's preference
        }
      );

      // Extract response
      const assistantMessage = response.choices[0]?.message;
      if (!assistantMessage) {
        throw new Error('No response from server');
      }

      // Decode audio if present
      let audioUrl: string | undefined;
      if (assistantMessage.audio?.data) {
        audioUrl = apiService.decodeAudioBase64(assistantMessage.audio.data);
      }

      // Add assistant message to chat
      const assistantMsg = {
        role: 'assistant' as const,
        content: assistantMessage.content,
        audioData: audioUrl,
        timestamp: Date.now(),
      };

      addMessage(chatId, assistantMsg);
    } catch (error: any) {
      // Check if it's a connection error and update server status
      if (error.message?.includes('connect') || error.message?.includes('No response')) {
        setServerStatus('offline');
      }

      // Add error message
      const errorMsg = {
        role: 'assistant' as const,
        content: `Error: ${error.message || 'Failed to get response from server'}`,
        timestamp: Date.now(),
      };
      addMessage(chatId, errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-dark-bg">
      {/* Mobile menu button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="fixed top-4 left-4 z-50 p-2 bg-dark-surface hover:bg-dark-surfaceHover rounded-lg md:hidden"
        aria-label="Toggle sidebar"
      >
        {sidebarOpen ? (
          <X className="w-5 h-5" />
        ) : (
          <Menu className="w-5 h-5" />
        )}
      </button>

      {/* Sidebar */}
      <ChatSidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(false)}
      />

      {/* Main content */}
      <div className="flex-1 flex flex-col md:ml-0">
        {/* Server status bar */}
        {serverStatus === 'offline' && (
          <div className="bg-dark-error/20 border-b border-dark-error/50 px-4 py-2">
            <p className="text-sm text-dark-error text-center">
              ⚠️ Server offline. Make sure the Omni server is running on port 8665
            </p>
          </div>
        )}

        {/* Chat area */}
        <ChatArea chat={currentChat} isLoading={isLoading} />

        {/* Input area */}
        <MessageInput 
          onSend={handleSend} 
          isLoading={isLoading} 
          disabled={serverStatus === 'offline'}
          audioOutputEnabled={audioOutputEnabled}
          onAudioOutputToggle={() => setAudioOutputEnabled(!audioOutputEnabled)}
        />
      </div>
    </div>
  );
}

