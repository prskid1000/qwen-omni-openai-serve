import { useState, useEffect, useCallback, useMemo } from 'react';
import { storage, Chat } from '../utils/storage';

export function useChatHistory() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    const loadedChats = storage.getChats();
    setChats(loadedChats);
    // Don't auto-select - let user choose or create a new chat
  }, []); // Only run once on mount

  const loadChats = useCallback(() => {
    const loadedChats = storage.getChats();
    setChats(loadedChats);
  }, []);

  const createNewChat = useCallback((): string => {
    const newChat: Chat = {
      id: storage.generateChatId(),
      title: 'New Chat',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    
    storage.saveChat(newChat);
    // Immediately add to chats array and set as current
    setChats(prev => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
    return newChat.id;
  }, []);

  // Compute current chat reactively from the chats array
  const currentChat = useMemo((): Chat | null => {
    if (!currentChatId) return null;
    // Get from chats array first (most up-to-date), fallback to storage
    const chatFromArray = chats.find(c => c.id === currentChatId);
    if (chatFromArray) return chatFromArray;
    // If not in chats array, try loading from storage
    const chatFromStorage = storage.getChat(currentChatId);
    if (chatFromStorage) {
      // Add to chats array if found in storage but not in array
      setChats(prev => {
        if (!prev.find(c => c.id === currentChatId)) {
          return [...prev, chatFromStorage];
        }
        return prev;
      });
      return chatFromStorage;
    }
    return null;
  }, [currentChatId, chats]); // Update when currentChatId or chats change

  const updateChat = useCallback((chatId: string, updates: Partial<Chat>) => {
    const chat = storage.getChat(chatId);
    if (!chat) return;

    const updatedChat: Chat = {
      ...chat,
      ...updates,
      updatedAt: Date.now(),
    };

    // Auto-generate title from first message if title is still "New Chat"
    if (updatedChat.messages.length > 0 && updatedChat.title === 'New Chat') {
      const firstUserMessage = updatedChat.messages.find(m => m.role === 'user');
      if (firstUserMessage) {
        updatedChat.title = storage.generateChatTitle(firstUserMessage.content);
      }
    }

    storage.saveChat(updatedChat);
    loadChats();
  }, [loadChats]);

  const addMessage = useCallback((chatId: string, message: Chat['messages'][0]) => {
    const chat = storage.getChat(chatId);
    if (!chat) return;

    const updatedChat: Chat = {
      ...chat,
      messages: [...chat.messages, message],
      updatedAt: Date.now(),
    };

    // Auto-generate title from first message
    if (updatedChat.messages.length === 1 && updatedChat.title === 'New Chat') {
      if (message.role === 'user') {
        updatedChat.title = storage.generateChatTitle(message.content);
      }
    }

    storage.saveChat(updatedChat);
    // Update both storage and state
    setChats(prev => {
      const index = prev.findIndex(c => c.id === chatId);
      if (index >= 0) {
        const updated = [...prev];
        updated[index] = updatedChat;
        return updated;
      }
      // If not found, add it
      return [updatedChat, ...prev];
    });
    loadChats(); // Also reload from storage to ensure sync
  }, [loadChats]);

  const deleteChat = useCallback((chatId: string) => {
    storage.deleteChat(chatId);
    loadChats();
    if (currentChatId === chatId) {
      setCurrentChatId(null);
    }
  }, [currentChatId, loadChats]);

  const renameChat = useCallback((chatId: string, newTitle: string) => {
    const chat = storage.getChat(chatId);
    if (!chat) return;
    
    const updatedChat: Chat = {
      ...chat,
      title: newTitle.trim() || 'New Chat',
      updatedAt: Date.now(),
    };
    
    storage.saveChat(updatedChat);
    loadChats();
  }, [loadChats]);

  const clearAllChats = useCallback(() => {
    storage.clearAll();
    loadChats();
    setCurrentChatId(null);
  }, [loadChats]);

  const filteredChats = chats.filter(chat => 
    chat.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    chat.messages.some(m => 
      m.content.toLowerCase().includes(searchQuery.toLowerCase())
    )
  );

  return {
    chats: filteredChats,
    currentChatId,
    currentChat,
    searchQuery,
    setCurrentChatId,
    setSearchQuery,
    createNewChat,
    updateChat,
    addMessage,
    deleteChat,
    renameChat,
    clearAllChats,
    loadChats,
  };
}

