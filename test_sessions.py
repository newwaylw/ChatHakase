#!/usr/bin/env python3
"""
Test script for chat session functionality.
This script demonstrates how the session manager works without requiring Slack.
"""

import asyncio
import time
from session_manager import SessionManager, MessageRole

def test_session_manager():
    """Test the session manager functionality."""
    print("🧪 Testing ChatHakase Session Manager\n")
    
    # Create a session manager
    manager = SessionManager(session_timeout=60)  # 1 minute timeout for testing
    
    # Simulate a conversation in channel1, thread1
    channel1 = "C1234567890"
    thread1 = "1234567890.123456"
    user1 = "U1111111111"
    
    print("1. Creating first session...")
    session1 = manager.get_or_create_session(channel1, thread1)
    print(f"   Session ID: {session1.session_id}")
    
    # Add some messages
    print("\n2. Adding messages to session...")
    session1.add_message(MessageRole.USER.value, "Hello, can you help me with Python?", user1, "Alice")
    session1.add_message(MessageRole.ASSISTANT.value, "Of course! I'd be happy to help you with Python. What specific topic would you like to learn about?")
    session1.add_message(MessageRole.USER.value, "I want to learn about async/await", user1, "Alice")
    
    print(f"   Session now has {len(session1.messages)} messages")
    
    # Test OpenAI message format
    print("\n3. Testing OpenAI message format...")
    openai_messages = session1.get_openai_messages()
    print(f"   Generated {len(openai_messages)} messages for OpenAI:")
    for i, msg in enumerate(openai_messages):
        content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"   {i+1}. {msg['role']}: {content_preview}")
    
    # Create another session in different thread
    print("\n4. Creating second session...")
    thread2 = "1234567890.654321"
    session2 = manager.get_or_create_session(channel1, thread2)
    session2.add_message(MessageRole.USER.value, "What's the weather like?", "U2222222222", "Bob")
    
    print(f"   Session 2 ID: {session2.session_id}")
    print(f"   Session 2 has {len(session2.messages)} messages")
    
    # Test session retrieval
    print("\n5. Testing session retrieval...")
    retrieved_session = manager.get_session(channel1, thread1)
    if retrieved_session:
        print(f"   Successfully retrieved session with {len(retrieved_session.messages)} messages")
    else:
        print("   ❌ Failed to retrieve session")
    
    # Test session stats
    print("\n6. Session statistics...")
    stats = manager.get_session_stats()
    print(f"   Total sessions: {stats['total_sessions']}")
    print(f"   Total messages: {stats['total_messages']}")
    print("   Session summaries:")
    for summary in stats['sessions']:
        print(f"   • {summary}")
    
    # Test session timeout (simulate)
    print("\n7. Testing session cleanup...")
    # Manually set old timestamp to test cleanup
    session2.last_activity = time.time() - 120  # 2 minutes ago
    cleaned = manager.cleanup_expired_sessions()
    print(f"   Cleaned up {cleaned} expired sessions")
    
    final_stats = manager.get_session_stats()
    print(f"   Remaining sessions: {final_stats['total_sessions']}")
    
    print("\n✅ Session manager tests completed!")

def test_conversation_flow():
    """Test a realistic conversation flow."""
    print("\n🗣️  Testing Conversation Flow\n")
    
    manager = SessionManager()
    channel = "C9876543210"
    thread = "9876543210.111111"
    user = "U3333333333"
    
    # Simulate a multi-turn conversation
    conversations = [
        ("What is machine learning?", "user"),
        ("Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.", "assistant"),
        ("Can you give me an example?", "user"),
        ("Sure! A common example is email spam detection. The system learns from thousands of emails labeled as 'spam' or 'not spam' to automatically identify spam in new emails.", "assistant"),
        ("How does it learn from the data?", "user"),
        ("Great question! Machine learning algorithms find patterns in the training data. For spam detection, it might learn that emails with certain words, sender patterns, or formatting are more likely to be spam.", "assistant")
    ]
    
    session = manager.get_or_create_session(channel, thread)
    
    for i, (content, role) in enumerate(conversations):
        if role == "user":
            session.add_message(MessageRole.USER.value, content, user, "Charlie")
        else:
            session.add_message(MessageRole.ASSISTANT.value, content)
        
        print(f"{i+1}. {role.upper()}: {content[:60]}{'...' if len(content) > 60 else ''}")
    
    print(f"\nConversation complete! Session has {len(session.messages)} messages")
    
    # Show how OpenAI would see this conversation
    print("\nOpenAI conversation format:")
    openai_msgs = session.get_openai_messages(max_messages=10)
    for msg in openai_msgs:
        role_emoji = "🤖" if msg['role'] == 'system' else "👤" if msg['role'] == 'user' else "🤖"
        print(f"{role_emoji} {msg['role']}: {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}")

if __name__ == "__main__":
    test_session_manager()
    test_conversation_flow()
