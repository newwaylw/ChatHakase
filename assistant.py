import openai
from enum import Enum
import asyncio
import time
from session_manager import session_manager, MessageRole
import logging
import requests
import base64
from io import BytesIO
from PIL import Image
import os

logger = logging.getLogger(__name__)


class Roles(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


PROMPT = {
    Roles.SYSTEM: "You are a helpful assistant, skilled in explaining complex concepts with concise sentences.",
    Roles.ASSISTANT:"You are a honest and helpful assistant, you always answer to the best of your knowledge, you would rather say I don't know than making up inaccurate answers."
}


async def process_request_streaming(content, client, channel, timestamp, user_id=None, user_name=None, model_name="gpt-4o"):
    """
    Process request with streaming response that updates Slack message in real-time.
    Now includes session management for conversation context.
    
    Args:
        content: The user's message content
        client: Slack client for API calls
        channel: Slack channel ID
        timestamp: Message timestamp for updates (used as thread_ts)
        user_id: Slack user ID
        user_name: User's display name
        model_name: OpenAI model to use
    """
    message_ts = None
    thread_ts = timestamp  # Use the original message timestamp as thread identifier
    
    try:
        # Clean up expired sessions periodically
        if len(session_manager.sessions) > 15:  # Arbitrary threshold
            cleaned = session_manager.cleanup_expired_sessions()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired sessions")
        
        # Add user message to session
        session = session_manager.add_user_message(
            channel_id=channel,
            thread_ts=thread_ts,
            content=content,
            user_id=user_id,
            user_name=user_name
        )
        
        logger.info(f"Processing request in session {session.session_id} with {len(session.messages)} messages")
        
        # Post initial "thinking" message in thread
        response = await client.chat_postMessage(
            channel=channel,
            text="🤔 Thinking...",
            thread_ts=thread_ts
        )
        message_ts = response["ts"]
        
        # Get conversation history for OpenAI
        openai_messages = session.get_openai_messages(include_system=True, max_messages=20)
        
        logger.debug(f"Sending {len(openai_messages)} messages to OpenAI")
        
        # Start streaming from OpenAI with conversation context
        stream = openai.Client().chat.completions.create(
            model=model_name,
            messages=openai_messages,
            stream=True,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Initialize response tracking
        full_response = ""
        last_update_time = time.time()
        update_interval = 0.5  # Update every 500ms to avoid rate limits
        chunk_count = 0
        
        # Process streaming chunks
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                chunk_count += 1
                
                # Update message periodically to show progress
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    try:
                        await client.chat_update(
                            channel=channel,
                            ts=message_ts,
                            text=full_response + " ⏳"  # Add typing indicator
                        )
                        last_update_time = current_time
                    except Exception as e:
                        logger.warning(f"Error updating message: {e}")
                        # Continue processing even if update fails
        
        # Final update with complete response
        if full_response.strip():
            await client.chat_update(
                channel=channel,
                ts=message_ts,
                text=full_response
            )
            
            # Add assistant response to session
            session_manager.add_assistant_message(
                channel_id=channel,
                thread_ts=thread_ts,
                content=full_response
            )
            
            logger.info(f"✅ Streaming complete: {chunk_count} chunks, {len(full_response)} characters")
            logger.info(f"Session now has {len(session.messages)} total messages")
        else:
            error_msg = "Sorry, I couldn't generate a response. Please try again."
            await client.chat_update(
                channel=channel,
                ts=message_ts,
                text=error_msg
            )
            
    except openai.RateLimitError as e:
        error_msg = "I'm currently experiencing high demand. Please try again in a moment."
        logger.error(f"OpenAI rate limit error: {e}")
        await _post_error_message(client, channel, thread_ts, message_ts, error_msg)
        
    except openai.APIError as e:
        error_msg = "I'm having trouble connecting to my AI service. Please try again."
        logger.error(f"OpenAI API error: {e}")
        await _post_error_message(client, channel, thread_ts, message_ts, error_msg)
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an unexpected error: {str(e)}"
        logger.error(f"Unexpected error in streaming response: {e}")
        await _post_error_message(client, channel, thread_ts, message_ts, error_msg)


async def _post_error_message(client, channel, thread_ts, message_ts, error_msg):
    """Helper function to post error messages with proper fallback."""
    try:
        if message_ts:
            # Update existing message
            await client.chat_update(
                channel=channel,
                ts=message_ts,
                text=error_msg
            )
        else:
            # Post new message in thread
            await client.chat_postMessage(
                channel=channel,
                text=error_msg,
                thread_ts=thread_ts
            )
    except Exception as post_error:
        logger.error(f"Error posting error message: {post_error}")


async def get_session_info(channel_id: str, thread_ts: str) -> str:
    """
    Get information about the current chat session.
    
    Args:
        channel_id: Slack channel ID
        thread_ts: Thread timestamp
        
    Returns:
        String with session information
    """
    session = session_manager.get_session(channel_id, thread_ts)
    
    if not session:
        return "No active session found for this thread."
    
    return session.get_summary()


async def clear_session(channel_id: str, thread_ts: str) -> str:
    """
    Clear the chat session for a specific thread.
    
    Args:
        channel_id: Slack channel ID
        thread_ts: Thread timestamp
        
    Returns:
        Confirmation message
    """
    session_id = session_manager._generate_session_id(channel_id, thread_ts)
    
    if session_id in session_manager.sessions:
        del session_manager.sessions[session_id]
        return f"✅ Chat session cleared for this thread."
    else:
        return "No active session found for this thread."


def get_all_sessions_info() -> str:
    """Get information about all active sessions."""
    stats = session_manager.get_session_stats()
    
    if stats["total_sessions"] == 0:
        return "No active chat sessions."
    
    info = f"📊 **Active Sessions**: {stats['total_sessions']}\n"
    info += f"💬 **Total Messages**: {stats['total_messages']}\n\n"
    
    for session_summary in stats["sessions"]:
        info += f"• {session_summary}\n"
    
    return info


async def process_deep_research_streaming(content, client, channel, timestamp, user_id=None, user_name=None):
    """
    Process request with OpenAI's deep research model.
    This uses the o1-mini model for complex research queries with enhanced reasoning.
    
    Args:
        content: The user's research query
        client: Slack client for API calls
        channel: Slack channel ID
        timestamp: Message timestamp for updates (used as thread_ts)
        user_id: Slack user ID
        user_name: User's display name
    """
    message_ts = None
    thread_ts = timestamp
    
    try:
        # Clean up expired sessions periodically
        if len(session_manager.sessions) > 15:
            cleaned = session_manager.cleanup_expired_sessions()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired sessions")
        
        # Add user message to session with deep research indicator
        session = session_manager.add_user_message(
            channel_id=channel,
            thread_ts=thread_ts,
            content=f"[DEEP RESEARCH] {content}",
            user_id=user_id,
            user_name=user_name
        )
        
        logger.info(f"Processing deep research request in session {session.session_id}")
        
        # Post initial "deep research" message in thread
        response = await client.chat_postMessage(
            channel=channel,
            text="🔬 Starting deep research analysis...",
            thread_ts=thread_ts
        )
        message_ts = response["ts"]
        
        logger.debug(f"Sending deep research request to o1-mini")
        
        # Create enhanced user message for deep research (o1-mini doesn't support system messages)
        enhanced_content = f"""Please provide a comprehensive, well-researched response with detailed analysis, multiple perspectives, and thorough explanations. Include relevant context, implications, and connections to related topics.

Research Query: {content}"""
        
        research_messages = [
            {
                "role": "user", 
                "content": enhanced_content
            }
        ]
        
        # Use the standard chat completions API with o1-mini model
        openai_client = openai.Client()
        
        try:
            # Try o1-mini first
            stream = openai_client.chat.completions.create(
                model="o4-mini-deep-research-2025-06-26",
                messages=research_messages,
                stream=True
            )
        except openai.APIError as e:
            if "model" in str(e).lower():
                # Fallback to GPT-4 if o1-mini is not available
                logger.info("o4-mini-deep-research-2025-06-26 not available, falling back to GPT-4")
                research_messages = [
                    {
                        "role": "system",
                        "content": "You are a deep research assistant. Provide comprehensive, well-researched responses with detailed analysis, multiple perspectives, and thorough explanations. Include relevant context, implications, and connections to related topics."
                    },
                    {
                        "role": "user", 
                        "content": content
                    }
                ]
                stream = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=research_messages,
                    stream=True,
                    max_tokens=4000,
                    temperature=0.3
                )
            else:
                raise e
        
        # Initialize response tracking
        full_response = ""
        last_update_time = time.time()
        update_interval = 1.0  # Update every 1 second for deep research
        chunk_count = 0
        
        # Process streaming chunks
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content_chunk = chunk.choices[0].delta.content
                full_response += content_chunk
                chunk_count += 1
                
                # Update message periodically to show progress
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    try:
                        # Show progress with research indicator
                        progress_text = full_response + "\n\n🔬 *Deep research in progress...*"
                        await client.chat_update(
                            channel=channel,
                            ts=message_ts,
                            text=progress_text
                        )
                        last_update_time = current_time
                    except Exception as e:
                        logger.warning(f"Error updating deep research message: {e}")
        
        # Final update with complete response
        if full_response.strip():
            # Add a header to indicate this was a deep research response
            final_response = f"🔬 **Deep Research Results**\n\n{full_response}"
            
            await client.chat_update(
                channel=channel,
                ts=message_ts,
                text=final_response
            )
            
            # Add assistant response to session
            session_manager.add_assistant_message(
                channel_id=channel,
                thread_ts=thread_ts,
                content=final_response
            )
            
            logger.info(f"✅ Deep research complete: {chunk_count} chunks, {len(full_response)} characters")
            logger.info(f"Session now has {len(session.messages)} total messages")
        else:
            error_msg = "Sorry, I couldn't complete the deep research analysis. Please try again with a more specific query."
            await client.chat_update(
                channel=channel,
                ts=message_ts,
                text=error_msg
            )
            
    except openai.RateLimitError as e:
        error_msg = "🔬 Deep research is currently experiencing high demand. Please try again in a moment."
        logger.error(f"OpenAI rate limit error in deep research: {e}")
        await _post_error_message(client, channel, thread_ts, message_ts, error_msg)
        
    except openai.APIError as e:
        error_msg = "🔬 I'm having trouble connecting to the deep research service. Please try again."
        logger.error(f"OpenAI API error in deep research: {e}")
        await _post_error_message(client, channel, thread_ts, message_ts, error_msg)
        
    except Exception as e:
        error_msg = f"🔬 Sorry, I encountered an error during deep research: {str(e)}"
        logger.error(f"Unexpected error in deep research: {e}")
        await _post_error_message(client, channel, thread_ts, message_ts, error_msg)


def process_request_non_streaming(content, role="user", model_name="gpt-4o"):
    """
    Legacy non-streaming function for backward compatibility.
    Note: This doesn't use session management - use process_request_streaming instead.
    
    Args:
        content: The message content
        role: The role for the message (default: "user")
        model_name: OpenAI model to use (default: "gpt-4o")
    
    Returns:
        String response from OpenAI
    """
    try:
        response = openai.Client().chat.completions.create(
            model=model_name,
            messages=[{"role": role, "content": content}],
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in non-streaming response: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


# async def download_slack_file(file_url: str, token: str) -> bytes:
#     """
#     Download a file from Slack using the bot token.
#
#     Args:
#         file_url: The URL of the file to download
#         token: Slack bot token for authentication
#
#     Returns:
#         File content as bytes
#     """
#     headers = {
#         'Authorization': f'Bearer {token}'
#     }
#
#     try:
#         # Use requests with proper error handling and redirect following
#         response = requests.get(file_url, headers=headers, allow_redirects=True, timeout=30)
#         response.raise_for_status()
#
#         # Log response details for debugging
#         logger.debug(f"Downloaded file: {len(response.content)} bytes, Content-Type: {response.headers.get('content-type', 'unknown')}")
#
#         # Verify we got actual file content, not a redirect page
#         content_type = response.headers.get('content-type', '').lower()
#         if 'text/html' in content_type and len(response.content) < 10000:
#             # This might be an error page or redirect, log the content for debugging
#             logger.warning(f"Received HTML response instead of file content: {response.text[:500]}")
#             raise Exception(f"Received HTML response instead of file content. Content-Type: {content_type}")
#
#         # Ensure we return bytes, not string
#         if isinstance(response.content, str):
#             logger.warning("Response content is string, converting to bytes")
#             return response.content.encode('utf-8')
#
#         return response.content
#
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Error downloading file from {file_url}: {e}")
#         raise Exception(f"Failed to download file: {str(e)}")
#     except Exception as e:
#         logger.error(f"Unexpected error downloading file: {e}")
#         raise


def debug_file_info(file_info: dict) -> None:
    """
    Debug helper to log file information structure.
    
    Args:
        file_info: Slack file object
    """
    logger.debug("=== FILE INFO DEBUG ===")
    logger.debug(f"File name: {file_info.get('name', 'N/A')}")
    logger.debug(f"File size: {file_info.get('size', 'N/A')} bytes")
    logger.debug(f"MIME type: {file_info.get('mimetype', 'N/A')}")
    logger.debug(f"File type: {file_info.get('filetype', 'N/A')}")
    
    # Log all URL-related fields
    url_fields = [k for k in file_info.keys() if 'url' in k.lower()]
    logger.debug(f"Available URL fields: {url_fields}")
    
    for field in url_fields:
        logger.debug(f"  {field}: {file_info.get(field, 'N/A')}")
    
    # Log other potentially useful fields
    other_fields = ['id', 'created', 'timestamp', 'mode', 'editable', 'is_external', 'external_type']
    for field in other_fields:
        if field in file_info:
            logger.debug(f"{field}: {file_info[field]}")
    
    logger.debug("=== END FILE INFO DEBUG ===")


def get_best_file_url(file_info: dict) -> str:
    """
    Get the best URL for downloading a file from Slack.
    
    Args:
        file_info: Slack file object
        
    Returns:
        Best URL to use for downloading
    """
    # Debug the file info structure
    debug_file_info(file_info)
    
    # Priority order for URLs to try
    url_fields = [
        'url_private_download',  # Best for downloading
        'url_private',           # Standard private URL
        'permalink_public',      # Public permalink (if available)
        'permalink',             # Fallback permalink
    ]
    
    for field in url_fields:
        if field in file_info and file_info[field]:
            logger.info(f"Using {field} for file download: {file_info[field]}")
            return file_info[field]
    
    # If none of the preferred fields exist, raise an error
    available_fields = [k for k in file_info.keys() if 'url' in k.lower()]
    logger.error(f"No suitable download URL found in file info. Available URL fields: {available_fields}")
    logger.error(f"Full file info keys: {list(file_info.keys())}")
    raise Exception(f"No suitable download URL found in file info. Available URL fields: {available_fields}")


async def download_slack_image(image_url: str, token: str) -> bytes:
    """
    Download an image from Slack using proper authentication and error handling.

    Args:
        image_url: The URL of the image to download
        token: Slack bot token for authentication

    Returns:
        Image content as bytes

    Raises:
        ValueError: If the URL is empty or invalid
        requests.RequestException: If the download fails
    """
    if not image_url or not isinstance(image_url, str) or not image_url.strip():
        raise ValueError("image_url must be a non-empty string")
    
    if not token:
        raise ValueError("token must be provided")

    # Prepare headers for Slack API authentication
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': 'ChatHakase-Bot/1.0',
        'Accept': 'image/*,*/*'
    }
    
    try:
        logger.debug(f"Downloading image from: {image_url}")
        
        # Use requests with proper error handling and redirect following
        response = requests.get(
            image_url.strip(), 
            headers=headers, 
            allow_redirects=True, 
            timeout=30,
            stream=True  # Stream to handle large files better
        )
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Get the content
        content = response.content
        
        # Log response details for debugging
        content_type = response.headers.get('content-type', '').lower()
        logger.debug(f"Downloaded file: {len(content)} bytes, Content-Type: {content_type}")
        logger.debug(f"Response status: {response.status_code}")
        logger.debug(f"Final URL after redirects: {response.url}")
        
        # Verify we got actual file content, not an HTML error page
        if 'text/html' in content_type:
            # This is likely an error page or redirect, log the content for debugging
            content_preview = content.decode('utf-8', errors='ignore')[:500] if content else "No content"
            logger.error(f"Received HTML response instead of image content: {content_preview}")
            raise Exception(f"Received HTML response instead of image content. Content-Type: {content_type}")
        
        # Check if content looks like an image (basic validation)
        if len(content) < 100:  # Images are typically larger than 100 bytes
            logger.warning(f"Downloaded content seems too small for an image: {len(content)} bytes")
        
        # Check for common image file signatures
        if content:
            # JPEG starts with FF D8 FF
            # PNG starts with 89 50 4E 47
            # GIF starts with 47 49 46 38
            # WEBP starts with 52 49 46 46 (RIFF) followed by WEBP
            if not (content.startswith(b'\xff\xd8\xff') or  # JPEG
                   content.startswith(b'\x89PNG') or        # PNG
                   content.startswith(b'GIF8') or           # GIF
                   (content.startswith(b'RIFF') and b'WEBP' in content[:20])):  # WEBP
                logger.warning(f"Content doesn't appear to be a valid image format. First 20 bytes: {content[:20]}")
        
        return content
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error downloading image from {image_url}: {e}")
        logger.error(f"Response status: {e.response.status_code if e.response else 'Unknown'}")
        logger.error(f"Response headers: {dict(e.response.headers) if e.response else 'Unknown'}")
        if e.response and e.response.content:
            error_content = e.response.content.decode('utf-8', errors='ignore')[:500]
            logger.error(f"Error response content: {error_content}")
        raise Exception(f"HTTP {e.response.status_code if e.response else 'Unknown'} error downloading image: {str(e)}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error downloading image from {image_url}: {e}")
        raise Exception(f"Failed to download image: {str(e)}")
        
    except Exception as e:
        logger.error(f"Unexpected error downloading image: {e}")
        raise


def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string for OpenAI API.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode('utf-8')


def resize_image_if_needed(image_bytes: bytes, max_size: int = 2048) -> tuple[bytes, str]:
    """
    Resize image if it's too large for OpenAI API.
    OpenAI Vision API supports: JPEG, PNG, GIF, and WEBP formats.
    
    Args:
        image_bytes: Raw image bytes
        max_size: Maximum dimension in pixels
        
    Returns:
        Tuple of (resized image bytes, mime_type)
    """
    # Validate input
    if not isinstance(image_bytes, bytes):
        raise ValueError(f"Expected bytes, got {type(image_bytes)}. Content: {str(image_bytes)[:200]}...")
    
    if len(image_bytes) == 0:
        raise ValueError("Received empty image data")
    
    try:
        image = Image.open(BytesIO(image_bytes))
        logger.debug(f"Processing image: format={image.format}, mode={image.mode}, size={image.size}")
        
        # Determine the original format and MIME type
        original_format = image.format
        if original_format == 'JPEG' or original_format == 'JPG':
            mime_type = 'image/jpeg'
            # Always use JPEG as the format name for consistency
            original_format = 'JPEG'
        elif original_format == 'PNG':
            mime_type = 'image/png'
        elif original_format == 'GIF':
            mime_type = 'image/gif'
        elif original_format == 'WEBP':
            mime_type = 'image/webp'
        else:
            # Convert unsupported formats to JPEG
            logger.info(f"Converting unsupported format {original_format} to JPEG")
            original_format = 'JPEG'
            mime_type = 'image/jpeg'
            # Convert RGBA to RGB for JPEG
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = background
        
        # Check if resize is needed
        if max(image.size) <= max_size:
            # Still need to convert format if necessary
            if image.format != original_format or image.mode not in ('RGB', 'RGBA'):
                logger.debug(f"Converting image mode from {image.mode} to appropriate format")
                output = BytesIO()
                if original_format == 'JPEG' and image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(output, format=original_format, quality=95 if original_format == 'JPEG' else None)
                return output.getvalue(), mime_type
            return image_bytes, mime_type
        
        # Calculate new size maintaining aspect ratio
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        logger.info(f"Resizing image from {image.size} to {new_size}")
        
        # Resize image
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert mode if necessary for JPEG
        if original_format == 'JPEG' and resized_image.mode != 'RGB':
            resized_image = resized_image.convert('RGB')
        
        # Save to bytes
        output = BytesIO()
        resized_image.save(output, format=original_format, quality=95 if original_format == 'JPEG' else None)
        
        logger.debug(f"Image processed successfully: {len(output.getvalue())} bytes, {mime_type}")
        return output.getvalue(), mime_type
    except Exception as e:
        logger.warning(f"Error resizing image: {e}, attempting fallback conversion to JPEG")
        # Log the first few bytes to help debug
        logger.debug(f"Image bytes preview: {image_bytes[:50] if len(image_bytes) >= 50 else image_bytes}")
        
        # Fallback: try to convert to JPEG
        try:
            image = Image.open(BytesIO(image_bytes))
            logger.info(f"Fallback conversion: format={image.format}, mode={image.mode}")
            
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            output = BytesIO()
            image.save(output, format='JPEG', quality=95)
            logger.info(f"Fallback conversion successful: {len(output.getvalue())} bytes")
            return output.getvalue(), 'image/jpeg'
        except Exception as fallback_error:
            logger.error(f"Fallback image conversion failed: {fallback_error}")
            # Last resort: return original bytes and hope for the best
            return image_bytes, 'image/jpeg'


async def process_image_with_text_streaming(content: str, image_files: list, client, channel: str, timestamp: str, 
                                          user_id: str = None, user_name: str = None, model_name: str = "gpt-4o", image_detail="low"):
    """
    Process request with both text and images using streaming response.
    
    Args:
        content: The user's text message
        image_files: List of image file objects from Slack
        client: Slack client for API calls
        channel: Slack channel ID
        timestamp: Message timestamp for updates
        user_id: Slack user ID
        user_name: User's display name
        model_name: OpenAI model to use (must support vision)
    """
    message_ts = None
    thread_ts = timestamp
    
    try:
        # Clean up expired sessions periodically
        if len(session_manager.sessions) > 15:
            cleaned = session_manager.cleanup_expired_sessions()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired sessions")
        
        # Add user message to session (text only for session history)
        session = session_manager.add_user_message(
            channel_id=channel,
            thread_ts=thread_ts,
            content=f"[IMAGE ANALYSIS] {content}",
            user_id=user_id,
            user_name=user_name
        )
        
        logger.info(f"Processing image analysis request in session {session.session_id} with {len(image_files)} images")
        
        # Post initial "analyzing" message in thread
        response = await client.chat_postMessage(
            channel=channel,
            text="🖼️ Analyzing images...",
            thread_ts=thread_ts
        )
        message_ts = response["ts"]
        
        # Download and process images
        processed_images = []
        slack_token = os.environ.get("SLACK_BOT_TOKEN")
        
        for i, file_info in enumerate(image_files):
            try:
                # Log file info for debugging
                logger.info(f"Processing image {i+1}/{len(image_files)}: {file_info.get('name', 'unknown')}")
                logger.info(f"Image MIME type: {file_info.get('mimetype', 'unknown')}")
                logger.info(f"Image file extension: {os.path.splitext(file_info.get('name', ''))[1]}")
                # Get the best URL for downloading the image
                file_url = get_best_file_url(file_info)
                logger.info(f"file download url: {file_url}")
                
                # Download image from Slack using improved method
                image_bytes = await download_slack_image(file_url, slack_token)
                
                # Debug: Check what we actually got
                logger.debug(f"Downloaded content type: {type(image_bytes)}")
                logger.debug(f"Downloaded content length: {len(image_bytes) if hasattr(image_bytes, '__len__') else 'N/A'}")
                
                if not isinstance(image_bytes, bytes):
                    logger.error(f"Expected bytes but got {type(image_bytes)}: {str(image_bytes)[:200]}...")
                    raise Exception(f"Downloaded content is {type(image_bytes)} instead of bytes")
                
                if len(image_bytes) == 0:
                    raise Exception("Downloaded image is empty (0 bytes)")
                
                # Resize if needed and get proper MIME type
                resized_bytes, mime_type = resize_image_if_needed(image_bytes)
                
                # Encode to base64
                base64_image = encode_image_to_base64(resized_bytes)
                
                processed_images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": image_detail
                    }
                })
                
                logger.info(f"Successfully processed image {i+1}/{len(image_files)}: {file_info.get('name', 'unknown')} ({mime_type})")
                
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {e}")
                await client.chat_update(
                    channel=channel,
                    ts=message_ts,
                    text=f"❌ Error processing image {i+1}: {str(e)}"
                )
                return
        
        # Prepare message content with images
        message_content = [
            {
                "type": "text",
                "text": content if content.strip() else "Please analyze these images."
            }
        ]
        message_content.extend(processed_images)
        
        # Get conversation history (text only)
        openai_messages = session.get_openai_messages(include_system=True, max_messages=10)
        
        # Add current message with images
        openai_messages.append({
            "role": "user",
            "content": message_content
        })
        
        logger.debug(f"Sending vision request to OpenAI with {len(processed_images)} images")
        
        # Start streaming from OpenAI with vision
        stream = openai.Client().chat.completions.create(
            model=model_name,
            messages=openai_messages,
            stream=True,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Initialize response tracking
        full_response = ""
        last_update_time = time.time()
        update_interval = 0.5
        chunk_count = 0
        
        # Process streaming chunks
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                chunk_count += 1
                
                # Update message periodically
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    try:
                        await client.chat_update(
                            channel=channel,
                            ts=message_ts,
                            text=full_response + " 🖼️"
                        )
                        last_update_time = current_time
                    except Exception as e:
                        logger.warning(f"Error updating vision message: {e}")
        
        # Final update with complete response
        if full_response.strip():
            await client.chat_update(
                channel=channel,
                ts=message_ts,
                text=full_response
            )
            
            # Add assistant response to session
            session_manager.add_assistant_message(
                channel_id=channel,
                thread_ts=thread_ts,
                content=full_response
            )
            
            logger.info(f"✅ Vision analysis complete: {chunk_count} chunks, {len(full_response)} characters")
        else:
            error_msg = "Sorry, I couldn't analyze the images. Please try again."
            await client.chat_update(
                channel=channel,
                ts=message_ts,
                text=error_msg
            )
            
    except Exception as e:
        error_msg = f"Sorry, I encountered an error analyzing the images: {str(e)}"
        logger.error(f"Error in image analysis: {e}")
        await _post_error_message(client, channel, thread_ts, message_ts, error_msg)


async def generate_image_streaming(prompt: str, client, channel: str, timestamp: str, 
                                 user_id: str = None, user_name: str = None, 
                                 size: str = "1024x1024", quality: str = "standard"):
    """
    Generate an image using DALL-E and post it to Slack.
    
    Args:
        prompt: Text description for image generation
        client: Slack client for API calls
        channel: Slack channel ID
        timestamp: Message timestamp for updates
        user_id: Slack user ID
        user_name: User's display name
        size: Image size (1024x1024, 1792x1024, or 1024x1792)
        quality: Image quality (standard or hd)
    """
    message_ts = None
    thread_ts = timestamp
    
    try:
        # Add user message to session
        session = session_manager.add_user_message(
            channel_id=channel,
            thread_ts=thread_ts,
            content=f"[IMAGE GENERATION] {prompt}",
            user_id=user_id,
            user_name=user_name
        )
        
        logger.info(f"Processing image generation request in session {session.session_id}")
        
        # Post initial "generating" message
        response = await client.chat_postMessage(
            channel=channel,
            text="🎨 Generating image...",
            thread_ts=thread_ts
        )
        message_ts = response["ts"]
        
        # Generate image with DALL-E
        openai_client = openai.Client()
        
        logger.debug(f"Generating image with prompt: {prompt}")
        
        image_response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1
        )
        
        image_url = image_response.data[0].url
        revised_prompt = getattr(image_response.data[0], 'revised_prompt', prompt)
        
        # Update message with generated image
        result_text = f"🎨 **Generated Image**\n\n**Prompt:** {prompt}"
        if revised_prompt != prompt:
            result_text += f"\n**Revised Prompt:** {revised_prompt}"
        
        # Post the image
        await client.chat_update(
            channel=channel,
            ts=message_ts,
            text=result_text,
            attachments=[
                {
                    "fallback": f"Generated image: {prompt}",
                    "image_url": image_url,
                    "title": "Generated Image",
                    "title_link": image_url
                }
            ]
        )
        
        # Add assistant response to session
        session_manager.add_assistant_message(
            channel_id=channel,
            thread_ts=thread_ts,
            content=f"Generated image: {prompt}"
        )
        
        logger.info(f"✅ Image generation complete for prompt: {prompt}")
        
    except openai.RateLimitError as e:
        error_msg = "🎨 Image generation is currently experiencing high demand. Please try again in a moment."
        logger.error(f"OpenAI rate limit error in image generation: {e}")
        await _post_error_message(client, channel, thread_ts, message_ts, error_msg)
        
    except openai.APIError as e:
        error_msg = "🎨 I'm having trouble connecting to the image generation service. Please try again."
        logger.error(f"OpenAI API error in image generation: {e}")
        await _post_error_message(client, channel, thread_ts, message_ts, error_msg)
        
    except Exception as e:
        error_msg = f"🎨 Sorry, I encountered an error generating the image: {str(e)}"
        logger.error(f"Error in image generation: {e}")
        await _post_error_message(client, channel, thread_ts, message_ts, error_msg)


