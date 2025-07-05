"""
Agent Communication Implementation

Handles communication between agents in EBAIF.
"""

import asyncio
from typing import Dict, List, Any, Optional
from ..utils import Logger

class AgentCommunication:
    """Handles inter-agent communication."""
    
    def __init__(self):
        self.message_queue: Dict[str, List[Dict[str, Any]]] = {}
        self.communication_history: List[Dict[str, Any]] = []
        self.logger = Logger.get_logger("AgentCommunication")
        
    async def send_message(self, sender_id: str, receiver_id: str, 
                          message: Dict[str, Any]):
        """Send a message between agents."""
        if receiver_id not in self.message_queue:
            self.message_queue[receiver_id] = []
            
        full_message = {
            'sender_id': sender_id,
            'receiver_id': receiver_id,
            'content': message,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        self.message_queue[receiver_id].append(full_message)
        self.communication_history.append(full_message)
        
        # Keep history manageable
        if len(self.communication_history) > 1000:
            self.communication_history = self.communication_history[-500:]
            
    async def get_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get pending messages for an agent."""
        if agent_id in self.message_queue:
            messages = self.message_queue[agent_id].copy()
            self.message_queue[agent_id].clear()
            return messages
        return []
        
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        total_messages = len(self.communication_history)
        
        # Count messages by sender
        sender_counts = {}
        for msg in self.communication_history:
            sender = msg['sender_id']
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
            
        return {
            'total_messages': total_messages,
            'sender_counts': sender_counts,
            'pending_messages': sum(len(queue) for queue in self.message_queue.values())
        }