import { useState } from 'react';
import { Menu, X } from 'lucide-react';
import { ChatArea } from './ChatArea';
import { MessageInput } from './MessageInput';
import { ChatSidebar } from './ChatSidebar';
import { useChatHistoryContext } from '../contexts/ChatHistoryContext';
import { apiService } from '../services/apiService';

// Helper function to convert File to base64 data URL
function convertFileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export function ChatContainer() {
  const [isLoading, setIsLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState<'online' | 'offline'>('online');
  const [audioOutputEnabled, setAudioOutputEnabled] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  const {
    currentChat,
    createNewChat,
    addMessage,
  } = useChatHistoryContext();

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
    let chatId = currentChat?.id;
    if (!chatId) {
      chatId = createNewChat();
    }

    setIsLoading(true);

    try {
      // Convert image/video files to base64 data URLs for persistence
      let imageDataUrl: string | undefined;
      let videoDataUrl: string | undefined;

      if (imageFile) {
        imageDataUrl = await convertFileToDataUrl(imageFile);
      }
      if (videoFile) {
        videoDataUrl = await convertFileToDataUrl(videoFile);
      }

      // Add user message to chat
      const userMessage = {
        role: 'user' as const,
        content: text || '[Media message]',
        timestamp: Date.now(),
        imageUrl: imageDataUrl,
        videoUrl: videoDataUrl,
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

      // Store base64 audio data (not blob URL) so it persists across reloads
      let audioData: string | undefined;
      if (assistantMessage.audio?.data) {
        // Store the base64 data directly, not the blob URL
        audioData = assistantMessage.audio.data;
      }

      // Add assistant message to chat
      const assistantMsg = {
        role: 'assistant' as const,
        content: assistantMessage.content,
        audioData: audioData, // Store base64, convert to blob URL when displaying
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
      {/* Sidebar */}
      <ChatSidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header with menu button */}
        <div className="flex items-center gap-4 p-4 border-b border-dark-border md:hidden">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-dark-surfaceHover rounded-lg transition-colors"
            aria-label="Toggle sidebar"
          >
            {sidebarOpen ? (
              <X className="w-5 h-5 text-dark-text" />
            ) : (
              <Menu className="w-5 h-5 text-dark-text" />
            )}
          </button>
          <h1 className="text-lg font-semibold text-dark-text">Omni Chat</h1>
        </div>

        {/* Server status bar */}
        {serverStatus === 'offline' && (
          <div className="bg-dark-error/20 border-b border-dark-error/50 px-4 py-2">
            <p className="text-sm text-dark-error text-center">
              ⚠️ Server offline. Make sure the Omni server is running on port 8665
            </p>
          </div>
        )}

        {/* Chat area */}
        <ChatArea key={currentChat?.id || 'no-chat'} chat={currentChat} isLoading={isLoading} />

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

