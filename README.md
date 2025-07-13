# ChatHakase
A chatbot based on Slack API and OpenAI API with streaming responses and conversation memory

## Features
- 🧵 **Thread-based Chat Sessions**: Each Slack thread maintains its own conversation context
- ⚡ **Real-time Streaming**: Live message updates as the AI generates responses
- 🧠 **Conversation Memory**: Bot remembers previous messages within each thread
- 🔄 **Multi-user Support**: Multiple users can participate in the same thread conversation
- 🛠️ **Session Management**: Automatic session creation, tracking, and cleanup
- 📊 **Admin Commands**: Session info, clearing, and statistics
- 🚨 **Error Handling**: Robust error handling and fallback mechanisms
- 🔬 **Deep Research Mode**: Advanced AI analysis using OpenAI's deep research model

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

### Special Commands
- `session info` - Show current session details
- `clear session` - Reset conversation history for this thread
- `all sessions` - Show all active sessions (admin)
- `help` - Show available commands

## Testing
- **Session Management**: Run `python test_sessions.py` to test session functionality
- **OpenAI Streaming**: Run `python test_streaming.py` to test OpenAI streaming without Slack
- **Deep Research**: Run `python test_deep_research.py` to test deep research functionality

## Architecture

### Core Components
- **`app.py`**: Main Slack application with event handlers
- **`assistant.py`**: OpenAI integration with streaming responses
- **`session_manager.py`**: Chat session management and conversation memory
- **`test_sessions.py`**: Test suite for session functionality

### How Streaming Works
- The bot posts an initial "thinking" message
- Updates the message every 500ms with partial responses
- Shows a typing indicator (⏳) during generation
- Finalizes with the complete response
- Stores the conversation in session memory

### Session Management
- Sessions are identified by `channel_id + thread_timestamp`
- Each session maintains conversation history
- OpenAI receives full context for better responses
- Automatic cleanup prevents memory leaks

## Documentation
- **[Chat Sessions Guide](CHAT_SESSIONS.md)**: Detailed documentation about session functionality
- **[Streaming Improvements](STREAMING_IMPROVEMENTS.md)**: Technical details about streaming implementation

## Development

### Project Structure
```
ChatHakase/
├── app.py                 # Main Slack application
├── assistant.py           # OpenAI integration
├── session_manager.py     # Session management
├── test_sessions.py       # Session tests
├── test_streaming.py      # Streaming tests
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
├── README.md             # This file
├── CHAT_SESSIONS.md      # Session documentation
└── STREAMING_IMPROVEMENTS.md
```

### Key Features Implementation
- **Thread Detection**: Automatically detects if message is in existing thread
- **Context Management**: Maintains up to 20 recent messages per session
- **Token Management**: Limits message history to stay within OpenAI token limits
- **Error Recovery**: Graceful handling of API failures and rate limits
- **Memory Efficiency**: Automatic cleanup of expired sessions

## Future Enhancements
- Persistent storage for conversation history
- Conversation summarization for long threads
- User-specific session limits
- Analytics and usage tracking
- Integration with other AI models
