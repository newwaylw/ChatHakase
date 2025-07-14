# ChatHakase
A chatbot based on Slack API and OpenAI API with streaming responses, conversation memory, and image capabilities

## Features
- 🧵 **Thread-based Chat Sessions**: Each Slack thread maintains its own conversation context
- ⚡ **Real-time Streaming**: Live message updates as the AI generates responses
- 🧠 **Conversation Memory**: Bot remembers previous messages within each thread
- 🔄 **Multi-user Support**: Multiple users can participate in the same thread conversation
- 🛠️ **Session Management**: Automatic session creation, tracking, and cleanup
- 📊 **Admin Commands**: Session info, clearing, and statistics
- 🚨 **Error Handling**: Robust error handling and fallback mechanisms
- 🔬 **Deep Research Mode**: Advanced AI analysis using OpenAI's deep research model
- 🖼️ **Image Analysis**: Upload and analyze images using GPT-4 Vision
- 🎨 **Image Generation**: Generate images using DALL-E 3

## How Chat Sessions Work
- **New Conversation**: Mention the bot or send a DM to start a new chat session
- **Continue Conversation**: Reply in the same thread to maintain context
- **Multiple Threads**: Each thread is a separate conversation with its own memory
- **Automatic Cleanup**: Sessions expire after 1 hour of inactivity

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment variables in `.env` (see `.env.example`)
3. Run the app: `python app.py`

## Environment Variables
```bash
# Slack Bot Configuration
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
```

## Usage

### Basic Chat
```
@ChatHakase Hello, can you help me with Python?
> Of course! I'd be happy to help you with Python. What would you like to learn?

What are decorators?
> [Remembers you're asking about Python] Decorators in Python are...
```

### Deep Research Mode
```
@ChatHakase /deep What are the latest developments in quantum computing and their potential impact on cryptography?
> 🔬 Deep Research Results
> 
> [Comprehensive analysis with detailed research findings...]
```

### Image Analysis
```
[Upload an image with text]
@ChatHakase What do you see in this image?
> 🖼️ [Analyzes the uploaded image and provides detailed description]
```

### Image Generation
```
@ChatHakase /generate a sunset over mountains with a lake in the foreground
> 🎨 Generating image...
> [Posts generated image]

@ChatHakase /image a cute robot assistant
> 🎨 [Generates and posts the requested image]
```

### Special Commands
- `session info` - Show current session details
- `clear session` - Reset conversation history for this thread
- `all sessions` - Show all active sessions (admin)
- `help` - Show available commands

## Testing
- **Session Management**: Run `python test_sessions.py` to test session functionality
- **OpenAI Streaming**: Run `python test_streaming.py` to test OpenAI streaming without Slack
- **Deep Research**: Run `python test_deep_research.py` to test deep research functionality
- **Image Features**: Run `python test_image_features.py` to test image processing and generation

## Architecture

### Core Components
- **`app.py`**: Main Slack application with event handlers and image support
- **`assistant.py`**: OpenAI integration with streaming responses, vision, and image generation
- **`session_manager.py`**: Chat session management and conversation memory
- **`test_sessions.py`**: Test suite for session functionality
- **`test_image_features.py`**: Test suite for image functionality

### How Streaming Works
- The bot posts an initial "thinking" message
- Updates the message every 500ms with partial responses
- Shows a typing indicator (⏳) during generation
- Finalizes with the complete response
- Stores the conversation in session memory

### Image Processing
- **Vision Analysis**: Automatically detects uploaded images and analyzes them using GPT-4 Vision
- **Image Resizing**: Automatically resizes large images to meet OpenAI API requirements
- **Base64 Encoding**: Converts images to base64 format for API transmission
- **Error Handling**: Robust error handling for image processing failures

### Image Generation
- **DALL-E 3**: Uses OpenAI's latest image generation model
- **Quality Options**: Supports both standard and HD quality generation
- **Size Options**: Supports multiple image sizes (1024x1024, 1792x1024, 1024x1792)
- **Prompt Enhancement**: DALL-E 3 automatically enhances prompts for better results

### Session Management
- Sessions are identified by `channel_id + thread_timestamp`
- Each session maintains conversation history including image interactions
- OpenAI receives full context for better responses
- Automatic cleanup prevents memory leaks

## New Image Commands

### Image Analysis Commands
- Upload any image file to Slack and mention the bot
- Add text with your image to ask specific questions
- Supports JPEG, PNG, GIF, and other common image formats

### Image Generation Commands
- `/generate <description>` - Generate an image using DALL-E 3
- `/image <description>` - Alternative command for image generation
- Examples:
  - `/generate a futuristic cityscape at night`
  - `/image a cute cat wearing a space helmet`

## Documentation
- **[Chat Sessions Guide](CHAT_SESSIONS.md)**: Detailed documentation about session functionality
- **[Streaming Improvements](STREAMING_IMPROVEMENTS.md)**: Technical details about streaming implementation

## Development

### Project Structure
```
ChatHakase/
├── app.py                    # Main Slack application with image support
├── assistant.py              # OpenAI integration with vision and generation
├── session_manager.py        # Session management
├── test_sessions.py          # Session tests
├── test_streaming.py         # Streaming tests
├── test_image_features.py    # Image functionality tests
├── requirements.txt          # Dependencies (updated with image libraries)
├── .env.example             # Environment template
├── README.md                # This file
├── CHAT_SESSIONS.md         # Session documentation
└── STREAMING_IMPROVEMENTS.md
```

### Key Features Implementation
- **Thread Detection**: Automatically detects if message is in existing thread
- **Context Management**: Maintains up to 20 recent messages per session
- **Token Management**: Limits message history to stay within OpenAI token limits
- **Error Recovery**: Graceful handling of API failures and rate limits
- **Memory Efficiency**: Automatic cleanup of expired sessions
- **Image Processing**: Automatic image detection, resizing, and encoding
- **Vision Integration**: Seamless integration with GPT-4 Vision for image analysis
- **Image Generation**: DALL-E 3 integration with quality and size options

## Future Enhancements
- Persistent storage for conversation history
- Conversation summarization for long threads
- User-specific session limits
- Analytics and usage tracking
- Integration with other AI models
- Image editing and manipulation features
- Batch image processing
- Custom image generation styles
