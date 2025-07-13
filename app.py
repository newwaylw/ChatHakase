import os
import asyncio
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from dotenv import load_dotenv
from pathlib import Path
from assistant import (
    process_request_streaming, 
    process_deep_research_streaming,
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


@app.event("app_mention")
@app.event('message')
async def handle_mentions(event, client, payload):
    """
    Handle mentions and messages with streaming responses and session management.
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
        
        # Handle /deep command for deep research
        if text.strip().lower().startswith('/deep'):
            deep_query = text[5:].strip()  # Remove '/deep' prefix
            if not deep_query:
                await client.chat_postMessage(
                    channel=event["channel"],
                    text="Please provide a query after the /deep command. Example: `/deep What are the latest developments in quantum computing?`",
                    thread_ts=event.get('thread_ts', event['ts'])
                )
                await client.reactions_remove(
                    channel=event["channel"],
                    timestamp=event["ts"],
                    name="eyes",
                )
                return
            
            # Process with deep research
            from assistant import process_deep_research_streaming
            await process_deep_research_streaming(
                content=deep_query,
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
        
        # Skip empty messages
        if not text.strip():
            await client.reactions_remove(
                channel=event["channel"],
                timestamp=event["ts"],
                name="eyes",
            )
            return
        
        # Determine thread timestamp
        # If this is a reply in a thread, use the thread_ts
        # If this is a new message, use the message ts as the thread starter
        thread_ts = event.get('thread_ts', event['ts'])
        
        logger.info(f"Processing message from {user_name} in thread {thread_ts}")
        
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

**Session Commands:**
• `session info` - Show current session details
• `clear session` - Reset conversation history for this thread
• `all sessions` - Show all active sessions (admin)
• `help` - Show this help message

**Features:**
✨ Conversation memory within threads
🔄 Real-time streaming responses
🧵 Thread-based chat sessions
⚡ Automatic session cleanup
🔬 Deep research mode for complex queries
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