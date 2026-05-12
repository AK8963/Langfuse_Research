"""Simulated user agent for generating realistic conversation turns"""

from langfuse.openai import OpenAI
from langfuse import get_client, observe
from typing import Dict, List

class SimulatedUser:
    """
    Simulated user that generates contextual messages based on persona and scenario
    """
    
    def __init__(self, persona: str, scenario: str, model: str, ollama_config: Dict):
        """
        Initialize simulated user with persona and scenario
        
        Args:
            persona: User characteristics and behavior patterns
            scenario: Conversation context and goals
            model: Model name to use for message generation
            ollama_config: Configuration dict with base_url and api_key
        """
        self.model = model
        self.client = OpenAI(
            base_url=ollama_config['base_url'],
            api_key=ollama_config['api_key'],
        )
        self.persona = persona
        self.scenario = scenario
        self.conversation_history = []
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for simulated user behavior"""
        return f"""You are a user in the following situation:
{self.scenario}

You have these characteristics:
{self.persona}

Your goal is to have a natural, continuous conversation. Ask follow-up questions based on the assistant's responses. Be curious and engaged. Keep the conversation flowing naturally by building on previous exchanges."""
    
    @observe(name="user-query")
    def generate_message(self, is_first_turn: bool = False) -> str:
        """
        Generate the next user message based on conversation history
        
        Args:
            is_first_turn: Whether this is the first turn of the conversation
            
        Returns:
            Generated user message
        """
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history for context
            messages.extend(self.conversation_history)
            
            # Add prompt for generating next message
            if is_first_turn:
                prompt = "Start the conversation by naturally introducing your situation and asking your first question."
            else:
                prompt = "Based on the assistant's last response, continue the conversation naturally with a relevant follow-up question or comment. Build on what was discussed."
            
            messages.append({"role": "user", "content": prompt})
            
            # Generate user message
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            user_message = response.choices[0].message.content
            
            # Update trace with generated message
            get_client().update_current_trace(
                output=user_message
            )
            
            return user_message
        
        except Exception as e:
            return f"Error generating user message: {str(e)}"
    
    def update_history(self, user_message: str, assistant_message: str):
        """
        Update the simulated user's conversation history
        
        Args:
            user_message: The user's message
            assistant_message: The assistant's response
        """
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the full conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []