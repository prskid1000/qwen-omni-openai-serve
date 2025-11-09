import { useEffect } from 'react';
import { X } from 'lucide-react';

interface FilePreviewProps {
  file: File;
  type: 'image' | 'video' | 'audio';
  onRemove: () => void;
}

export function FilePreview({ file, type, onRemove }: FilePreviewProps) {
  const url = URL.createObjectURL(file);

  // Clean up object URL on unmount
  useEffect(() => {
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [url]);

  return (
    <div className="relative inline-block mt-2">
      {type === 'image' && (
        <div className="relative group">
          <img
            src={url}
            alt={file.name}
            className="max-w-xs max-h-48 rounded-lg object-cover"
          />
          <button
            onClick={onRemove}
            className="absolute top-2 right-2 p-1 bg-dark-surface/80 hover:bg-dark-surface rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
      
      {type === 'video' && (
        <div className="relative group">
          <video
            src={url}
            controls
            className="max-w-xs max-h-48 rounded-lg"
          />
          <button
            onClick={onRemove}
            className="absolute top-2 right-2 p-1 bg-dark-surface/80 hover:bg-dark-surface rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
      
      {type === 'audio' && (
        <div className="flex items-center gap-2 p-2 bg-dark-surface rounded-lg">
          <audio src={url} controls className="flex-1" />
          <button
            onClick={onRemove}
            className="p-1 hover:bg-dark-surfaceHover rounded-full transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}

