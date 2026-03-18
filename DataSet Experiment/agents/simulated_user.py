from langfuse.openai import OpenAI
from langfuse import get_client, observe
from typing import List, Dict
from utils.utils import logger, handle_error

class SimulatedUser:
    """Simulated user for conversation testing"""
    
    def __init__(self, persona: str, scenario: str, model: str, ollama_config: Dict):
        self.model = model
        self.client = OpenAI(
            base_url=ollama_config['base_url'],
            api_key=ollama_config['api_key']
        )
        self.persona = persona
        self.scenario = scenario
        self.conversation_history = []
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for simulated user"""
        return f"""You are a user in the following situation:
{self.scenario}

You have these characteristics:
{self.persona}

Your goal is to have a natural, continuous conversation. Ask follow-up questions based on the assistant's responses. Be curious and engaged. Keep the conversation flowing naturally by building on previous exchanges."""
    
    @observe(name="user-query")
    def generate_message(self, is_first_turn: bool = False) -> str:
        """Generate the next user message based on conversation history"""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history)
            
            if is_first_turn:
                prompt = "Start the conversation by naturally introducing your situation and asking your first question."
            else:
                prompt = "Based on the assistant's last response, continue the conversation naturally with a relevant follow-up question or comment. Build on what was discussed."
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            user_message = response.choices[0].message.content
            
            get_client().update_current_trace(output=user_message)
            
            return user_message
        
        except Exception as e:
            return handle_error(e, "simulated user message generation")
    
    def update_history(self, user_message: str, assistant_message: str):
        """Update the simulated user's conversation history"""
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})