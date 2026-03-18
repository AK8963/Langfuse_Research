from langfuse.openai import OpenAI
from langfuse import get_client, observe, propagate_attributes
from typing import List, Dict
from utils.utils import logger, handle_error

class LlamaChatbot:
    """AI Assistant chatbot with Langfuse tracing"""
    
    def __init__(self, model: str, ollama_config: Dict, session_id: str = None):
        self.model = model
        self.client = OpenAI(
            base_url=ollama_config['base_url'],
            api_key=ollama_config['api_key']
        )
        self.conversation_history = []
        self.session_id = session_id
    
    @observe(name="assistant-response")
    def chat(self, user_message: str, turn_number: int) -> str:
        """Single turn chat with session and trace tracking"""
        try:
            with propagate_attributes(session_id=self.session_id):
                self.conversation_history.append({
                    "role": "user",
                    "content": user_message
                })
                
                full_conversation = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions about dates, time calculations, and general inquiries accurately."
                    }
                ] + self.conversation_history
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_conversation
                )
                
                assistant_message = response.choices[0].message.content
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                get_client().update_current_trace(
                    name=f"Turn {turn_number}",
                    input=user_message,
                    output=assistant_message,
                    metadata={"turn": turn_number}
                )
                
                return assistant_message
        
        except Exception as e:
            return handle_error(e, f"chatbot turn {turn_number}")