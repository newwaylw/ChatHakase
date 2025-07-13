"""
Chat Session Manager for Slack-based OpenAI conversations.
Manages conversation history and context using Slack threads.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """Represents a single message in a chat session."""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    user_name: Optional[str] = None


@dataclass
class ChatSession:
    """Represents a chat session with conversation history."""
    session_id: str
    channel_id: str
    thread_ts: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    participants: set = field(default_factory=set)
    
    def add_message(self, role: str, content: str, user_id: str = None, user_name: str = None):
        """Add a message to the session."""
        message = ChatMessage(
            role=role,
            content=content,
            user_id=user_id,
            user_name=user_name
        )
        self.messages.append(message)
        self.last_activity = time.time()
        
        if user_id:
            self.participants.add(user_id)
    
    def get_openai_messages(self, include_system: bool = True, max_messages: int = 50) -> List[Dict]:
        """
        Convert session messages to OpenAI chat format.
        
        Args:
            include_system: Whether to include system messages
            max_messages: Maximum number of messages to include (for token management)
        
        Returns:
            List of messages in OpenAI format
        """
        openai_messages = []
        
        # Add system message if requested
        if include_system:
            openai_messages.append({
                "role": MessageRole.SYSTEM.value,
                "content": "You are a helpful assistant in a Slack conversation. "
                          "Maintain context from previous messages in this thread. "
                          "Be concise but thorough in your responses."
            })
        
        # Get recent messages (excluding system messages from history)
        recent_messages = [msg for msg in self.messages if msg.role != MessageRole.SYSTEM.value]
        recent_messages = recent_messages[-max_messages:]
        
        # Convert to OpenAI format
        for msg in recent_messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return openai_messages
    
    def get_summary(self) -> str:
        """Get a summary of the session."""
        user_count = len(self.participants)
        message_count = len(self.messages)
        duration = time.time() - self.created_at
        
        return (f"Session {self.session_id}: {message_count} messages, "
                f"{user_count} participants, {duration/60:.1f}min old")


class SessionManager:
    """Manages multiple chat sessions."""
    
    def __init__(self, session_timeout: int = 3600):  # 1 hour default timeout
        self.sessions: Dict[str, ChatSession] = {}
        self.session_timeout = session_timeout
        
    def _generate_session_id(self, channel_id: str, thread_ts: str) -> str:
        """Generate a unique session ID from channel and thread timestamp."""
        return f"{channel_id}_{thread_ts}"
    
    def get_or_create_session(self, channel_id: str, thread_ts: str) -> ChatSession:
        """
        Get existing session or create a new one.
        
        Args:
            channel_id: Slack channel ID
            thread_ts: Slack thread timestamp (use message ts for new threads)
        
        Returns:
            ChatSession object
        """
        session_id = self._generate_session_id(channel_id, thread_ts)
        
        if session_id not in self.sessions:
            logger.info(f"Creating new chat session: {session_id}")
            self.sessions[session_id] = ChatSession(
                session_id=session_id,
                channel_id=channel_id,
                thread_ts=thread_ts
            )
        else:
            # Update last activity
            self.sessions[session_id].last_activity = time.time()
            logger.debug(f"Retrieved existing session: {session_id}")
        
        return self.sessions[session_id]
    
    def add_user_message(self, channel_id: str, thread_ts: str, content: str, 
                        user_id: str, user_name: str = None) -> ChatSession:
        """Add a user message to the session."""
        session = self.get_or_create_session(channel_id, thread_ts)
        session.add_message(
            role=MessageRole.USER.value,
            content=content,
            user_id=user_id,
            user_name=user_name
        )
        logger.debug(f"Added user message to session {session.session_id}")
        return session
    
    def add_assistant_message(self, channel_id: str, thread_ts: str, content: str) -> ChatSession:
        """Add an assistant message to the session."""
        session = self.get_or_create_session(channel_id, thread_ts)
        session.add_message(
            role=MessageRole.ASSISTANT.value,
            content=content
        )
        logger.debug(f"Added assistant message to session {session.session_id}")
        return session
    
    def get_session(self, channel_id: str, thread_ts: str) -> Optional[ChatSession]:
        """Get an existing session without creating a new one."""
        session_id = self._generate_session_id(channel_id, thread_ts)
        return self.sessions.get(session_id)
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions to free memory."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            logger.info(f"Cleaning up expired session: {session_id}")
            del self.sessions[session_id]
        
        return len(expired_sessions)
    
    def get_session_stats(self) -> Dict:
        """Get statistics about active sessions."""
        total_sessions = len(self.sessions)
        total_messages = sum(len(session.messages) for session in self.sessions.values())
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "sessions": [session.get_summary() for session in self.sessions.values()]
        }


# Global session manager instance
session_manager = SessionManager()
