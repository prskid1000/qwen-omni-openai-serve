import { useEffect, useRef } from 'react';
import { MessageBubble } from './MessageBubble';
import { Loader2 } from 'lucide-react';
import { Chat } from '../utils/storage';

interface ChatAreaProps {
  chat: Chat | null;
  isLoading?: boolean;
}

export function ChatArea({ chat, isLoading = false }: ChatAreaProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Auto-scroll to bottom when new messages arrive in the same chat
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chat?.messages, chat?.id]); // Include chat.id to detect chat switches

  useEffect(() => {
    // Scroll to top when switching to a different chat
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = 0;
    }
  }, [chat?.id]); // Scroll to top when chat ID changes

  if (!chat) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-semibold mb-2">Welcome to Omni Chat</h2>
          <p className="text-dark-textSecondary">
            Start a new conversation to begin chatting with Qwen2.5-Omni
          </p>
        </div>
      </div>
    );
  }

  return (
    <div ref={chatAreaRef} className="flex-1 overflow-y-auto scrollbar-thin p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        {chat.messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <h2 className="text-2xl font-semibold mb-2">{chat.title}</h2>
              <p className="text-dark-textSecondary">
                Send a message to start the conversation
              </p>
            </div>
          </div>
        ) : (
          <>
            {chat.messages
              .filter((message) => message.role !== 'system') // Filter out system messages
              .map((message, index) => (
                <MessageBubble
                  key={`${message.timestamp}-${index}`}
                  role={message.role}
                  content={message.content}
                  audioUrl={message.audioData}
                  imageUrl={message.imageUrl}
                  videoUrl={message.videoUrl}
                  timestamp={message.timestamp}
                />
              ))}
            
            {isLoading && (
              <div className="flex justify-start mb-4">
                <div className="bg-dark-surface rounded-2xl px-4 py-3 shadow-lg">
                  <Loader2 className="w-5 h-5 animate-spin text-dark-accent" />
                </div>
              </div>
            )}
          </>
        )}
        
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}

