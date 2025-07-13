import openai
from enum import Enum
import asyncio
import time
from session_manager import session_manager, MessageRole
import logging

logger = logging.getLogger(__name__)


class Roles(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


PROMPT = {
    Roles.SYSTEM: "You are a helpful assistant, skilled in explaining complex concepts with concise sentences.",
    Roles.ASSISTANT:"You are a honest and helpful assistant, you always answer to the best of your knowledge, you would rather say I don't know than making up inaccurate answers."
}


# def process_thread_with_assistant(user_query, assistant_id, model="gpt-4-1106-preview", from_user=None):
#     """
#     Process a thread with an assistant and handle the response which includes text and images.
#
#     :param user_query: The user's query.
#     :param assistant_id: The ID of the assistant to be used.
#     :param model: The model version of the assistant.
#     :param from_user: The user ID from whom the query originated.
#     :return: A dictionary containing text responses and in-memory file objects.
#     """
#     response_texts = []  # List to store text responses
#     response_files = []  # List to store file IDs
#     in_memory_files = []  # List to store in-memory file objects
#
#     try:
#         print("Creating a thread for the user query...")
#         thread = openai.Client().beta.threads.create()
#         print(f"Thread created with ID: {thread.id}")
#
#         print("Adding the user query as a message to the thread...")
#         openai.Client().beta.threads.messages.create(
#             thread_id=thread.id,
#             role="user",
#             content=user_query
#         )
#         print("User query added to the thread.")
#
#         print("Creating a run to process the thread with the assistant...")
#         run = openai.Client().beta.threads.runs.create(
#             thread_id=thread.id,
#             assistant_id=assistant_id,
#             model=model
#         )
#         print(f"Run created with ID: {run.id}")
#
#         while True:
#             print("Checking the status of the run...")
#             run_status = openai.Client().beta.threads.runs.retrieve(
#                 thread_id=thread.id,
#                 run_id=run.id
#             )
#             print(f"Current status of the run: {run_status.status}")
#
#             if run_status.status == "requires_action":
#                 print("Run requires action. Executing specified function...")
#                 tool_call = run_status.required_action.submit_tool_outputs.tool_calls[0]
#                 function_name = tool_call.function.name
#                 arguments = json.loads(tool_call.function.arguments)
#
#                 function_output = execute_function(function_name, arguments, from_user)
#                 function_output_str = json.dumps(function_output)
#
#                 print("Submitting tool outputs...")
#                 openai.Client().beta.threads.runs.submit_tool_outputs(
#                     thread_id=thread.id,
#                     run_id=run.id,
#                     tool_outputs=[{
#                         "tool_call_id": tool_call.id,
#                         "output": function_output_str
#                     }]
#                 )
#                 print("Tool outputs submitted.")
#
#             elif run_status.status in ["completed", "failed", "cancelled"]:
#                 print("Fetching messages added by the assistant...")
#                 messages = openai.Client().beta.threads.messages.list(thread_id=thread.id)
#                 for message in messages.data:
#                     if message.role == "assistant":
#                         for content in message.content:
#                             if content.type == "text":
#                                 response_texts.append(content.text.value)
#                             elif content.type == "image_file":
#                                 file_id = content.image_file.file_id
#                                 response_files.append(file_id)
#
#                 print("Messages fetched. Retrieving content for each file ID...")
#                 for file_id in response_files:
#                     try:
#                         print(f"Retrieving content for file ID: {file_id}")
#                         # Retrieve file content from OpenAI API
#                         file_response = openai.Client().files.content(file_id)
#                         if hasattr(file_response, 'content'):
#                             # If the response has a 'content' attribute, use it as binary content
#                             file_content = file_response.content
#                         else:
#                             # Otherwise, use the response directly
#                             file_content = file_response
#
#                         in_memory_file = io.BytesIO(file_content)
#                         in_memory_files.append(in_memory_file)
#                         print(f"In-memory file object created for file ID: {file_id}")
#                     except Exception as e:
#                         print(f"Failed to retrieve content for file ID: {file_id}. Error: {e}")
#
#                 break
#             sleep(1)
#
#         return {"text": response_texts, "in_memory_files": in_memory_files}
#
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return {"text": [], "in_memory_files": []}


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
        if len(session_manager.sessions) > 10:  # Arbitrary threshold
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
        if len(session_manager.sessions) > 10:
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
                model="o1-mini",
                messages=research_messages,
                stream=True
            )
        except openai.APIError as e:
            if "model" in str(e).lower():
                # Fallback to GPT-4 if o1-mini is not available
                logger.info("o1-mini not available, falling back to GPT-4")
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


