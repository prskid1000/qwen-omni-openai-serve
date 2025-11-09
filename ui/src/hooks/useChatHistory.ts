import { useState, useEffect, useCallback } from 'react';
import { storage, Chat } from '../utils/storage';

export function useChatHistory() {
  const [currentChat, setCurrentChat] = useState<Chat | null>(null);

  // Load the most recent chat on mount
  useEffect(() => {
    const chats = storage.getChats();
    if (chats.length > 0) {
      // Get the most recently updated chat
      const mostRecent = chats.sort((a, b) => b.updatedAt - a.updatedAt)[0];
      setCurrentChat(mostRecent);
    }
  }, []);

  // Create a new chat
  const createNewChat = useCallback((): string => {
    const newChat: Chat = {
      id: storage.generateChatId(),
      title: 'New Chat',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    
    // Save to storage
    storage.saveChat(newChat);
    
    // Set as current chat
    setCurrentChat(newChat);
    
    return newChat.id;
  }, []);

  // Add a message to a chat
  const addMessage = useCallback((chatId: string, message: Chat['messages'][0]) => {
    // Get chat from storage (source of truth)
    const chat = storage.getChat(chatId) || currentChat;
    if (!chat || chat.id !== chatId) {
      console.error(`Chat ${chatId} not found`);
      return;
    }

    // Create updated chat
    const updatedChat: Chat = {
      ...chat,
      messages: [...chat.messages, message],
      updatedAt: Date.now(),
    };

    // Auto-generate title from first user message
    if (updatedChat.messages.length === 1 && updatedChat.title === 'New Chat') {
      if (message.role === 'user') {
        updatedChat.title = storage.generateChatTitle(message.content);
      }
    }

    // Save to storage
    storage.saveChat(updatedChat);

    // Update current chat state
    if (currentChat && currentChat.id === chatId) {
      setCurrentChat(updatedChat);
    }
  }, [currentChat]);

  return {
    currentChat,
    createNewChat,
    addMessage,
  };
}
