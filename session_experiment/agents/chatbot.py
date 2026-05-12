"""LLM Chatbot with session tracking for multi-turn conversations"""

from langfuse.openai import OpenAI
from langfuse import get_client, observe, propagate_attributes
from typing import Dict, List

class LlamaChatbot:
    """
    Chatbot that maintains conversation history and tracks sessions
    """
    
    def __init__(self, model: str, ollama_config: Dict, session_id: str = None):
        """
        Initialize chatbot with model configuration and session tracking
        
        Args:
            model: Model name to use for responses
            ollama_config: Configuration dict with base_url and api_key
            session_id: Session identifier for grouping traces
        """
        self.model = model
        self.client = OpenAI(
            base_url=ollama_config['base_url'],
            api_key=ollama_config['api_key'],
        )
        self.conversation_history = []
        self.session_id = session_id
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the assistant"""
        return "You are a helpful assistant that answers questions about dates, time calculations, and general inquiries accurately."
    
    @observe(name="assistant-response")
    def chat(self, user_message: str, turn_number: int) -> str:
        """
        Generate assistant response for a single turn with session tracking
        
        Args:
            user_message: User's input message
            turn_number: Current turn number in conversation
            
        Returns:
            Assistant's response message
        """
        try:
            with propagate_attributes(session_id=self.session_id):
                # Add user message to history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_message
                })
                
                # Build full conversation context
                full_conversation = [
                    {"role": "system", "content": self.system_prompt}
                ] + self.conversation_history
                
                # Generate response
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_conversation
                )
                
                assistant_message = response.choices[0].message.content
                
                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                # Update trace metadata
                get_client().update_current_trace(
                    name=f"Turn {turn_number}",
                    input=user_message,
                    output=assistant_message,
                    metadata={"turn": turn_number}
                )
                
                return assistant_message
        
        except Exception as e:
            error_msg = f"Error in chatbot: {str(e)}"
            return error_msg
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the full conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []