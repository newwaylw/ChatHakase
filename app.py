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


# @app.command("/generate")
# async def handle_generate_command(ack, body, client, logger):
#     """
#     Handle /generate slash command for image generation.
#     """
#     await ack()
#
#     try:
#         # Get command details
#         user_id = body["user_id"]
#         channel_id = body["channel_id"]
#         text = body.get("text", "").strip()
#
#         # Get user info
#         response = await client.users_info(user=user_id)
#         user_name = response["user"].get("real_name", response["user"].get("display_name", "Unknown"))
#
#         if not text:
#             await client.chat_postEphemeral(
#                 channel=channel_id,
#                 user=user_id,
#                 text="Please provide a description after the /generate command. Example: `/generate a sunset over mountains`"
#             )
#             return
#
#         # Post initial message to start the thread
#         initial_response = await client.chat_postMessage(
#             channel=channel_id,
#             text=f"🎨 Generating image: {text}"
#         )
#
#         thread_ts = initial_response["ts"]
#
#         # Process image generation
#         await generate_image_streaming(
#             prompt=text,
#             client=client,
#             channel=channel_id,
#             timestamp=thread_ts,
#             user_id=user_id,
#             user_name=user_name
#         )
#
#     except Exception as e:
#         logger.error(f"Error handling /generate command: {e}")
#         await client.chat_postEphemeral(
#             channel=body["channel_id"],
#             user=body["user_id"],
#             text=f"Sorry, I encountered an error generating the image: {str(e)}"
#         )


@app.command("/image")
async def handle_image_command(ack, body, client, logger):
    """
    Handle /image slash command for both image generation and image processing.
    
    Usage:
    - /image <description> - Generate an image with DALL-E
    - /image <question> (with file upload) - Analyze uploaded images
    - /image analyze <question> (with file upload) - Explicitly analyze images
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
        
        # Check if there are any files in the recent messages (look back a few messages)
        # This is a workaround since slash commands don't directly include file uploads
        recent_files = await get_recent_uploaded_files(client, channel_id, user_id)
        
        # Parse command text to determine intent
        command_parts = text.lower().split()
        explicit_mode = None
        actual_text = text

        if command_parts and command_parts[0] in ['analyze', 'analyse', 'process', 'describe']:
            explicit_mode = 'analyze'
            actual_text = ' '.join(text.split()[1:])  # Remove the first word
        if command_parts and command_parts[0] in ['generate', 'create', 'make', 'draw']:
            explicit_mode = 'generate'
            actual_text = ' '.join(text.split()[1:])  # Remove the first word

        # Determine operation mode
        if explicit_mode == 'analyze' or (recent_files and not explicit_mode):
            # Image analysis mode
            if not recent_files:
                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text="🖼️ No recent images found. Please upload an image file and then use the `/image` command.\n\n" +
                         "**Examples:**\n" +
                         "• Upload an image, then: `/image What do you see in this image?`\n" +
                         "• Upload an image, then: `/image analyze the contents of this screenshot`"
                )
                return
            
            if not actual_text.strip():
                actual_text = "Please analyze this image and describe what you see."
            
            # Post initial message to start the thread
            initial_response = await client.chat_postMessage(
                channel=channel_id,
                text=f"🖼️ Analyzing uploaded image(s): {actual_text}"
            )
            
            thread_ts = initial_response["ts"]
            
            # Process image analysis
            await process_image_with_text_streaming(
                content=actual_text,
                image_files=recent_files,
                client=client,
                channel=channel_id,
                timestamp=thread_ts,
                user_id=user_id,
                user_name=user_name
            )
            
        else:
            # Image generation mode (default)
            if not actual_text.strip():
                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text="🎨 Please provide a description for image generation.\n\n" +
                         "**Examples:**\n" +
                         "• `/image a cute cat wearing a space helmet`\n" +
                         "• `/image generate a sunset over mountains with a lake`\n" +
                         "• `/image create a futuristic cityscape at night`\n\n" +
                         "**For image analysis:**\n" +
                         "• Upload an image first, then use `/image analyze <question>`"
                )
                return
            
            # Post initial message to start the thread
            initial_response = await client.chat_postMessage(
                channel=channel_id,
                text=f"🎨 Generating image: {actual_text}"
            )
            
            thread_ts = initial_response["ts"]
            
            # Process image generation
            await generate_image_streaming(
                prompt=actual_text,
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
        
        # # Handle /deep command for deep research
        # if text.strip().lower().startswith('/deep'):
        #     deep_query = text[5:].strip()  # Remove '/deep' prefix
        #     if not deep_query:
        #         await client.chat_postMessage(
        #             channel=event["channel"],
        #             text="Please provide a query after the /deep command. Example: `/deep What are the latest developments in quantum computing?`",
        #             thread_ts=event.get('thread_ts', event['ts'])
        #         )
        #         await client.reactions_remove(
        #             channel=event["channel"],
        #             timestamp=event["ts"],
        #             name="eyes",
        #         )
        #         return
        #
        #     # Process with deep research
        #     from assistant import process_deep_research_streaming
        #     await process_deep_research_streaming(
        #         content=deep_query,
        #         client=client,
        #         channel=event["channel"],
        #         timestamp=event.get('thread_ts', event['ts']),
        #         user_id=user_id,
        #         user_name=user_name
        #     )
        #
        #     # Remove eyes reaction when done
        #     await client.reactions_remove(
        #         channel=event["channel"],
        #         timestamp=event["ts"],
        #         name="eyes",
        #     )
        #     return
        
        # Handle /generate or /image command for image generation
        if text.strip().lower().startswith(('/generate', '/image')):
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
        
        # Check for uploaded files (images)
        files = event.get('files', [])
        image_files = []
        
        for file in files:
            # Check if file is an image
            if file.get('mimetype', '').startswith('image/'):
                image_files.append(file)
        
        # Determine thread timestamp
        thread_ts = event.get('thread_ts', event['ts'])
        
        # Handle image analysis if images are present
        if image_files:
            logger.info(f"Processing {len(image_files)} images from {user_name} in thread {thread_ts}")
            
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


async def get_recent_uploaded_files(client, channel_id: str, user_id: str, limit: int = 10) -> list:
    """
    Get recent uploaded image files from the channel by the user.
    
    Args:
        client: Slack client
        channel_id: Channel to search in
        user_id: User who uploaded the files
        limit: Number of recent messages to check
    
    Returns:
        List of image file objects
    """
    try:
        # Get recent messages from the channel
        response = await client.conversations_history(
            channel=channel_id,
            limit=limit
        )
        
        image_files = []
        
        # Look through recent messages for files uploaded by this user
        for message in response.get('messages', []):
            # Check if message is from the same user and has files
            if (message.get('user') == user_id and 
                message.get('files') and 
                # Only consider files from the last 5 minutes to avoid old uploads
                time.time() - float(message.get('ts', 0)) < 300):
                
                for file in message['files']:
                    # Check if file is an image
                    if file.get('mimetype', '').startswith('image/'):
                        image_files.append(file)
        
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
    logger.info("Starting ChatHakase with session management...")
    handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())