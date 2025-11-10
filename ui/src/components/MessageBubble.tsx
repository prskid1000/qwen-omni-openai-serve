import { AudioPlayer } from './AudioPlayer';
import { ToolCallDisplay } from './ToolCallDisplay';
import { ToolCall } from '../services/apiService';

interface MessageBubbleProps {
  role: 'user' | 'assistant' | 'tool';
  content: string;
  audioUrl?: string;
  imageUrl?: string;
  videoUrl?: string;
  timestamp: number;
  toolCalls?: ToolCall[];
  toolCallId?: string;
}

export function MessageBubble({
  role,
  content,
  audioUrl,
  imageUrl,
  videoUrl,
  timestamp,
  toolCalls,
  toolCallId,
}: MessageBubbleProps) {
  const isUser = role === 'user';
  const isTool = role === 'tool';
  const time = new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

  // Tool messages have a different style
  if (isTool) {
    return (
      <div className="flex w-full mb-4 justify-start">
        <div className="max-w-[80%] md:max-w-[70%] min-w-0 bg-dark-surfaceSecondary text-dark-text rounded-2xl px-4 py-3 shadow-lg border border-dark-border">
          <div className="text-xs text-dark-textSecondary mb-1">Tool Result</div>
          <div className="whitespace-pre-wrap break-words font-mono text-sm">
            {content}
          </div>
          <div className="text-xs mt-2 text-dark-textSecondary">
            {time}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`flex w-full mb-4 ${
        isUser ? 'justify-end' : 'justify-start'
      }`}
    >
      <div
        className={`max-w-[80%] md:max-w-[70%] min-w-0 ${
          isUser
            ? 'bg-dark-accent text-white'
            : 'bg-dark-surface text-dark-text'
        } rounded-2xl px-4 py-3 shadow-lg`}
      >
        {/* Text content */}
        {content && (
          <div className="whitespace-pre-wrap break-words mb-2">
            {content}
          </div>
        )}

        {/* Tool calls */}
        {toolCalls && toolCalls.length > 0 && (
          <ToolCallDisplay toolCalls={toolCalls} />
        )}

        {/* Media previews */}
        {imageUrl && (
          <div className="mb-2">
            <img
              src={imageUrl}
              alt="Uploaded"
              className="max-w-full rounded-lg"
            />
          </div>
        )}
        
        {videoUrl && (
          <div className="mb-2">
            <video src={videoUrl} controls className="max-w-full rounded-lg" />
          </div>
        )}

        {/* Audio player for assistant messages */}
        {!isUser && audioUrl && (
          <div className="mt-2">
            <AudioPlayer audioUrl={audioUrl} />
          </div>
        )}

        {/* Timestamp */}
        <div
          className={`text-xs mt-2 ${
            isUser ? 'text-white/70' : 'text-dark-textSecondary'
          }`}
        >
          {time}
        </div>
      </div>
    </div>
  );
}

