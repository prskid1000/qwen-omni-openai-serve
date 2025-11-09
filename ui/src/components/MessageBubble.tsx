import { AudioPlayer } from './AudioPlayer';

interface MessageBubbleProps {
  role: 'user' | 'assistant';
  content: string;
  audioUrl?: string;
  imageUrl?: string;
  videoUrl?: string;
  timestamp: number;
}

export function MessageBubble({
  role,
  content,
  audioUrl,
  imageUrl,
  videoUrl,
  timestamp,
}: MessageBubbleProps) {
  const isUser = role === 'user';
  const time = new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

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

