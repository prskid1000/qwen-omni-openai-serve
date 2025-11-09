# Omni Chat UI

A modern, professional React chat interface for the Qwen2.5-Omni Server API.

## Features

- 🎨 **Modern Dark Theme** - Sleek, professional design with Tailwind CSS
- 💬 **Chat History** - Persistent chat history with sidebar navigation
- 🎵 **Audio Playback** - Embedded audio player in each assistant response
- 📎 **Multimodal Input** - Support for text, audio, image, and video files
- 🎤 **Voice Recording** - Record audio directly in the browser
- 📱 **Responsive Design** - Works on desktop and mobile devices
- ⚡ **Fast & Smooth** - Built with Vite for optimal performance

## Prerequisites

- Node.js 18+ and npm
- Omni Server running on `http://localhost:8665`

## Installation

1. Install dependencies:
```bash
npm install
```

2. (Optional) Configure API URL in `.env`:
```
VITE_API_URL=http://localhost:8665
```

## Development

Start the development server:
```bash
npm run dev
```

The UI will be available at `http://localhost:3000`

## Build

Build for production:
```bash
npm run build
```

The built files will be in the `dist` directory.

## Usage

1. Make sure the Omni Server is running (`python -m app.main`)
2. Open the UI in your browser
3. Start a new chat or continue an existing conversation
4. Type messages, upload files, or record audio
5. Each assistant response includes an embedded audio player

## Project Structure

```
ui/
├── src/
│   ├── components/     # React components
│   ├── hooks/          # Custom React hooks
│   ├── services/       # API service
│   ├── utils/          # Utility functions
│   └── styles/         # CSS styles
├── public/             # Static assets
└── package.json        # Dependencies
```

## Technologies

- React 18
- TypeScript
- Vite
- Tailwind CSS
- Axios
- Lucide React (icons)

