from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, max_pairs: int = 5):
        # We track max_pairs (each pair contains 1 user message and 1 assistant reply)
        self.max_pairs = max_pairs
        # Map: session_id -> list of message dicts: {"role": "user"|"assistant", "content": str}
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def add_message(self, session_id: str, role: str, content: str):
        """Appends a message to the session's history and limits history to the last N message pairs."""
        if not session_id:
            return
            
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        self.sessions[session_id].append({"role": role, "content": content})
        
        # Bounding the history: each pair is 2 messages (User + Assistant)
        max_messages = self.max_pairs * 2
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]
            logger.info(f"Pruned history for session {session_id} to last {self.max_pairs} message pairs.")

    def get_history_string(self, session_id: str) -> str:
        """Formats the conversation history as a string for context inclusion."""
        if not session_id or session_id not in self.sessions or not self.sessions[session_id]:
            return "No previous history."
            
        formatted_messages = []
        for msg in self.sessions[session_id]:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            formatted_messages.append(f"{role_label}: {msg['content']}")
            
        return "\n".join(formatted_messages)

    def clear_session(self, session_id: str):
        """Resets the history for a given session ID."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session history for {session_id}")
