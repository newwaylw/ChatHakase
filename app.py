import os
import asyncio
import time
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from dotenv import load_dotenv
from pathlib import Path
from assistant import (
    process_request_streaming, 
    process_deep_research_streaming,
    process_image_with_text_streaming,
    generate_image_streaming,
    get_session_info, 
    clear_session, 
    get_all_sessions_info
)
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env_path = Path(".env")
load_dotenv(env_path)
app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))

# Global tracker for recent uploads
class RecentUploadsTracker:
    def __init__(self):
        self.recent_uploads = {}  # user_id -> list of file info
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def add_upload(self, user_id: str, channel_id: str, file_info: dict):
        """Add a file upload to the tracker."""
        if user_id not in self.recent_uploads:
            self.recent_uploads[user_id] = []
        
        file_info['timestamp'] = time.time()
        file_info['channel_id'] = channel_id
        self.recent_uploads[user_id].append(file_info)
        
        # Keep only the most recent 5 files per user
        self.recent_uploads[user_id] = self.recent_uploads[user_id][-5:]
        
        logger.info(f"Tracked upload: {file_info.get('name', 'unknown')} for user {user_id}")
        self._cleanup_if_needed()
    
    def get_recent_uploads(self, user_id: str, channel_id: str = None, max_age: int = 600) -> list:
        """Get recent uploads for a user."""
        self._cleanup_if_needed()
        
        if user_id not in self.recent_uploads:
            return []
        
        current_time = time.time()
        recent_files = []
        
        for file_info in self.recent_uploads[user_id]:
            age = current_time - file_info.get('timestamp', 0)
            
            if age <= max_age:
                if channel_id is None or file_info.get('channel_id') == channel_id:
                    recent_files.append(file_info)
        
        logger.info(f"Retrieved {len(recent_files)} recent uploads for user {user_id}")
        return recent_files
    
    def _cleanup_if_needed(self):
        """Clean up old entries if needed."""
        current_time = time.time()
        
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries()
            self.last_cleanup = current_time
    
    def _cleanup_old_entries(self):
        """Remove old entries from the tracker."""
        current_time = time.time()
        max_age = 900  # 15 minutes
        
        for user_id in list(self.recent_uploads.keys()):
            self.recent_uploads[user_id] = [
                file_info for file_info in self.recent_uploads[user_id]
                if current_time - file_info.get('timestamp', 0) <= max_age
            ]
            
            if not self.recent_uploads[user_id]:
                del self.recent_uploads[user_id]

upload_tracker = RecentUploadsTracker()


@app.command("/deep")
async def handle_deep_command(ack, body, client, logger):
    """
    Handle /deep slash command for deep research queries.
    """
    await ack()
    
    try:
        # Get command details
        user_id = body["user_id"]
        channel_id = body["channel_id"]
        text = body.get("text", "").strip()
        
        # Get user info
        response = await client.users_info(user=user_id)
        user_name = response["user"].get("real_name", response["user"].get("display_name", "Unknown"))
        
        if not text:
            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text="Please provide a query after the /deep command. Example: `/deep What are the latest developments in quantum computing?`"
            )
            return
        
        # Post initial message to start the thread
        initial_response = await client.chat_postMessage(
            channel=channel_id,
            text=f"🔬 Starting deep research for: {text}"
        )
        
        thread_ts = initial_response["ts"]
        
        # Process with deep research
        await process_deep_research_streaming(
            content=text,
            client=client,
            channel=channel_id,
            timestamp=thread_ts,
            user_id=user_id,
            user_name=user_name
        )
        
    except Exception as e:
        logger.error(f"Error handling /deep command: {e}")
        await client.chat_postEphemeral(
            channel=body["channel_id"],
            user=body["user_id"],
            text=f"Sorry, I encountered an error processing your deep research request: {str(e)}"
        )


@app.command("/image")
async def handle_image_command(ack, body, client, logger):
    """
    Handle /image slash command for both image generation and image processing.

    Usage:
    - /image <description> - Generate an image with DALL-E
    - /image <question> (with recent file upload) - Analyze uploaded images
    - /image analyze <question> (with recent file upload) - Explicitly analyze images
    - /image generate <description> - Explicitly generate an image
    """
    await ack()

    try:
        # Get command details
        user_id = body["user_id"]
        channel_id = body["channel_id"]
        text = body.get("text", "").strip()

        # Get user info
        response = await client.users_info(user=user_id)
        user_name = response["user"].get("real_name", response["user"].get("display_name", "Unknown"))

        # Image generation mode (default)
        if not text:
            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text="🎨 Please provide a description for image generation.\n\n" +
                     "**Examples:**\n" +
                     "• `/image a cute cat wearing a space helmet`\n" +
                     "• `/image generate a sunset over mountains with a lake`\n" +
                     "• `/image create a futuristic cityscape at night`\n\n"
            )
            return

        # Post initial message to start the thread
        initial_response = await client.chat_postMessage(
            channel=channel_id,
            text=f"🎨 Generating image: {text}"
        )

        thread_ts = initial_response["ts"]

        # Process image generation
        await generate_image_streaming(
            prompt=text,
            client=client,
            channel=channel_id,
            timestamp=thread_ts,
            user_id=user_id,
            user_name=user_name
        )

    except Exception as e:
        logger.error(f"Error handling /image command: {e}")
        await client.chat_postEphemeral(
            channel=body["channel_id"],
            user=body["user_id"],
            text=f"Sorry, I encountered an error processing your image request: {str(e)}"
        )


@app.event("app_mention")
@app.event('message')
async def handle_mentions(event, client, payload):
    """
    Handle mentions and messages with streaming responses, session management, and image support.
    """
    try:
        # Skip bot messages to avoid loops
        if event.get('bot_id'):
            return
            
        # Add eyes reaction to show we're processing
        await client.reactions_add(
            channel=event["channel"],
            timestamp=event["ts"],
            name="eyes",
        )
        
        # Get user info
        response = await client.users_info(user=event['user'])
        user_name = response["user"].get("real_name", response["user"].get("display_name", "Unknown"))
        user_id = event['user']
        text = event.get('text', '')
        
        # Remove bot mention from text if present
        bot_user_id = await client.auth_test()
        bot_mention = f"<@{bot_user_id['user_id']}>"
        if bot_mention in text:
            text = text.replace(bot_mention, "").strip()
        
        # Handle special commands
        if await handle_special_commands(text, client, event, user_name):
            # Remove eyes reaction when done with command
            await client.reactions_remove(
                channel=event["channel"],
                timestamp=event["ts"],
                name="eyes",
            )
            return

        # Check for uploaded files (images) and track them
        files = event.get('files', [])
        image_files = []

        for file in files:
            # Check if file is an image
            if file.get('mimetype', '').startswith('image/'):
                image_files.append(file)
                # Track this upload for later /image commands
                upload_tracker.add_upload(user_id, event["channel"], file)

        # Determine thread timestamp
        thread_ts = event.get('thread_ts', event['ts'])
        # image file present, assume we are dealing with image analysis requests
        if image_files:
            logger.info(f"image file {len(image_files)} present from {user_name} in thread {thread_ts}")

            await process_image_with_text_streaming(
                content=text,
                image_files=image_files,
                client=client,
                channel=event["channel"],
                timestamp=thread_ts,
                user_id=user_id,
                user_name=user_name
            )

            # Remove eyes reaction when done
            await client.reactions_remove(
                channel=event["channel"],
                timestamp=event["ts"],
                name="eyes",
            )
            return


        # Handle /generate or /image command for image generation
        elif text.strip().lower().startswith(('/generate', '/image')):
            # Extract command and prompt
            if text.strip().lower().startswith('/generate'):
                prompt = text[9:].strip()  # Remove '/generate' prefix
            else:
                prompt = text[6:].strip()  # Remove '/image' prefix

            if not prompt:
                await client.chat_postMessage(
                    channel=event["channel"],
                    text="Please provide a description after the command. Example: `/generate a sunset over mountains`",
                    thread_ts=event.get('thread_ts', event['ts'])
                )
                await client.reactions_remove(
                    channel=event["channel"],
                    timestamp=event["ts"],
                    name="eyes",
                )
                return
            
            # Process image generation
            await generate_image_streaming(
                prompt=prompt,
                client=client,
                channel=event["channel"],
                timestamp=event.get('thread_ts', event['ts']),
                user_id=user_id,
                user_name=user_name
            )

            # Remove eyes reaction when done
            await client.reactions_remove(
                channel=event["channel"],
                timestamp=event["ts"],
                name="eyes",
            )
            return

        # # Check for uploaded files (images) and track them
        # files = event.get('files', [])
        # image_files = []
        #
        # for file in files:
        #     # Check if file is an image
        #     if file.get('mimetype', '').startswith('image/'):
        #         image_files.append(file)
        #         # Track this upload for later /image commands
        #         upload_tracker.add_upload(user_id, event["channel"], file)
        #
        # # Determine thread timestamp
        # thread_ts = event.get('thread_ts', event['ts'])

        # Handle image analysis if images are present
        # if image_files:
        #     logger.info(f"Processing {len(image_files)} images from {user_name} in thread {thread_ts}")

            # await process_image_with_text_streaming(
            #     content=text,
            #     image_files=image_files,
            #     client=client,
            #     channel=event["channel"],
            #     timestamp=thread_ts,
            #     user_id=user_id,
            #     user_name=user_name
            # )
            #
            # # Remove eyes reaction when done
            # await client.reactions_remove(
            #     channel=event["channel"],
            #     timestamp=event["ts"],
            #     name="eyes",
            # )
            # return
        
        # Skip empty messages (no text and no images)
        if not text.strip():
            await client.reactions_remove(
                channel=event["channel"],
                timestamp=event["ts"],
                name="eyes",
            )
            return
        
        logger.info(f"Processing text message from {user_name} in thread {thread_ts}")
        
        # Process with streaming response and session management
        await process_request_streaming(
            content=text,
            client=client,
            channel=event["channel"],
            timestamp=thread_ts,
            user_id=user_id,
            user_name=user_name
        )

        # Remove eyes reaction when done
        await client.reactions_remove(
            channel=event["channel"],
            timestamp=event["ts"],
            name="eyes",
        )
        
    except Exception as e:
        logger.error(f"Error handling mention: {e}")
        try:
            await client.chat_postMessage(
                channel=event["channel"],
                text=f"Sorry, I encountered an error: {str(e)}",
                thread_ts=event.get('thread_ts', event['ts'])
            )
        except Exception as post_error:
            logger.error(f"Error posting error message: {post_error}")


async def get_recent_uploaded_files_improved(client, channel_id: str, user_id: str, limit: int = 20, time_window: int = 600) -> list:
    """
    Improved function to get recent uploaded image files from the channel by the user.
    """
    try:
        logger.info(f"Searching for recent files by user {user_id} in channel {channel_id}")
        
        # Get recent messages from the channel
        response = await client.conversations_history(
            channel=channel_id,
            limit=limit
        )
        
        image_files = []
        current_time = time.time()
        
        # Look through recent messages for files uploaded by this user
        for message in response.get('messages', []):
            message_time = float(message.get('ts', 0))
            time_diff = current_time - message_time
            
            # Check if message is from the same user and has files
            if (message.get('user') == user_id and 
                message.get('files') and 
                time_diff < time_window):  # Extended time window
                
                logger.info(f"Found message with files from target user (age: {time_diff:.1f}s)")
                
                for file in message['files']:
                    # Check if file is an image
                    if file.get('mimetype', '').startswith('image/'):
                        # Add additional metadata for debugging
                        file_info = {
                            **file,
                            'message_ts': message.get('ts'),
                            'upload_age_seconds': time_diff,
                            'message_text': message.get('text', '')
                        }
                        image_files.append(file_info)
                        logger.info(f"Found image file: {file.get('name', 'unknown')} ({file.get('mimetype', 'unknown')})")
        
        logger.info(f"Found {len(image_files)} recent image files via message history")
        return image_files
        
    except Exception as e:
        logger.error(f"Error getting recent uploaded files: {e}")
        return []


async def handle_special_commands(text: str, client, event, user_name: str) -> bool:
    """
    Handle special bot commands like session info, clear session, etc.
    
    Returns:
        True if a special command was handled, False otherwise
    """
    text_lower = text.lower().strip()
    channel = event["channel"]
    thread_ts = event.get('thread_ts', event['ts'])
    
    # Session info command
    if text_lower in ['session info', 'session status', 'info']:
        session_info = await get_session_info(channel, thread_ts)
        await client.chat_postMessage(
            channel=channel,
            text=f"📋 **Session Info**\n{session_info}",
            thread_ts=thread_ts
        )
        return True
    
    # Clear session command
    elif text_lower in ['clear session', 'reset session', 'clear', 'reset']:
        result = await clear_session(channel, thread_ts)
        await client.chat_postMessage(
            channel=channel,
            text=result,
            thread_ts=thread_ts
        )
        return True
    
    # All sessions info (admin command)
    elif text_lower in ['all sessions', 'sessions', 'admin info']:
        sessions_info = get_all_sessions_info()
        await client.chat_postMessage(
            channel=channel,
            text=sessions_info,
            thread_ts=thread_ts
        )
        return True
    
    # Debug uploads command (new)
    elif text_lower in ['debug uploads', 'show uploads', 'recent uploads']:
        user_id = event['user']
        tracked_files = upload_tracker.get_recent_uploads(user_id, channel, max_age=900)
        
        if tracked_files:
            debug_info = f"🔍 **Recent Uploads Debug**\n\n"
            for i, file_info in enumerate(tracked_files):
                age = time.time() - file_info.get('timestamp', 0)
                debug_info += f"**File {i+1}:**\n"
                debug_info += f"• Name: {file_info.get('name', 'unknown')}\n"
                debug_info += f"• Type: {file_info.get('mimetype', 'unknown')}\n"
                debug_info += f"• Age: {age:.1f} seconds\n"
                debug_info += f"• Channel: {file_info.get('channel_id', 'unknown')}\n\n"
        else:
            debug_info = "🔍 **Recent Uploads Debug**\n\nNo recent uploads found for your user ID."
        
        await client.chat_postMessage(
            channel=channel,
            text=debug_info,
            thread_ts=thread_ts
        )
        return True
    
    # Help command
    elif text_lower in ['help', 'commands']:
        help_text = """
🤖 **ChatHakase Commands**

**Chat Commands:**
• Just mention me or send a message to start chatting
• Continue the conversation in the same thread for context
• `/deep <query>` - Use deep research AI for complex questions

**Image Commands:**
• Upload images with text to analyze them 🖼️
• `/image <description>` - Generate images with DALL-E 🎨
• `/image analyze <question>` - Analyze recently uploaded images
• `/image generate <description>` - Generate images (explicit)
• `/generate <description>` - Alternative command for image generation

**Session Commands:**
• `session info` - Show current session details
• `clear session` - Reset conversation history for this thread
• `all sessions` - Show all active sessions (admin)
• `debug uploads` - Show recent file uploads (debug)
• `help` - Show this help message

**Image Command Examples:**
🎨 **Generation:**
• `/image a sunset over mountains`
• `/image generate a cute robot assistant`
• `/image create a futuristic cityscape`

🖼️ **Analysis:**
• Upload image → `/image what do you see?`
• Upload screenshot → `/image analyze this interface`
• Upload diagram → `/image explain this flowchart`

**Features:**
✨ Conversation memory within threads
🔄 Real-time streaming responses
🧵 Thread-based chat sessions
⚡ Automatic session cleanup
🔬 Deep research mode for complex queries
🖼️ Image analysis and vision capabilities
🎨 AI image generation with DALL-E

**Tips for Image Analysis:**
• Upload your image first, then use `/image` within 10 minutes
• Use `debug uploads` to see if your files were detected
• Mention me directly with images for immediate analysis
        """
        await client.chat_postMessage(
            channel=channel,
            text=help_text,
            thread_ts=thread_ts
        )
        return True
    
    return False


async def main():
    """
    Main async function to start the app.
    """
    logger.info("Starting ChatHakase with improved image upload detection...")
    handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
