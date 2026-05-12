import logging
import json
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.json") -> Dict[str, Any]:
    """Load configuration from JSON file with environment variable substitution"""
    try:
        load_dotenv()
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Substitute environment variables
        config['langfuse']['public_key'] = os.getenv('LANGFUSE_PUBLIC_KEY')
        config['langfuse']['secret_key'] = os.getenv('LANGFUSE_SECRET_KEY')
        config['langfuse']['host'] = os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
        
        logger.info("Configuration loaded successfully")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def validate_langfuse_connection(langfuse_client) -> bool:
    """Validate Langfuse authentication"""
    try:
        if not langfuse_client.auth_check():
            logger.error("Langfuse authentication failed")
            return False
        logger.info("Langfuse authentication successful")
        return True
    except Exception as e:
        logger.error(f"Langfuse connection error: {e}")
        return False

def safe_json_parse(json_string: str, default: Dict = None) -> Dict:
    """Safely parse JSON with error handling"""
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}. Returning default.")
        return default or {}

def log_turn_info(turn_number: int, user_msg: str, assistant_msg: str, max_length: int = 100):
    """Log conversation turn information"""
    logger.info(f"--- Turn {turn_number} ---")
    logger.info(f"USER: {user_msg[:max_length]}...")
    logger.info(f"ASSISTANT: {assistant_msg[:max_length]}...")

def log_session_info(session_id: str, num_turns: int, session_score: float = None):
    """Log session-level information"""
    logger.info(f"\n{'='*80}")
    logger.info(f"SESSION SUMMARY: {session_id}")
    logger.info(f"Total Turns: {num_turns}")
    if session_score is not None:
        logger.info(f"Session Quality Score: {session_score:.2f}")
    logger.info(f"{'='*80}\n")

def handle_error(error: Exception, context: str = "") -> str:
    """Centralized error handling"""
    error_msg = f"Error in {context}: {str(error)}"
    logger.error(error_msg, exc_info=True)
    return error_msg

class SessionManager:
    """Manage session IDs and conversation state for session-level evaluation"""
    
    @staticmethod
    def generate_session_id(item_id: str, prefix: str = "session") -> str:
        """
        Generate unique session ID for grouping multiple traces
        
        Args:
            item_id: Unique identifier from dataset item
            prefix: Prefix for session ID
            
        Returns:
            Session ID string for grouping related traces
        """
        session_id = f"{prefix}-{item_id}"
        logger.debug(f"Generated session ID: {session_id}")
        return session_id
    
    @staticmethod
    def generate_trace_id(langfuse_client, session_id: str, turn_number: int) -> str:
        """
        Generate deterministic trace ID for a specific turn within a session
        
        Uses create_trace_id() to generate a 32 hexchar lowercase string
        that is deterministic based on the seed value.
        
        Args:
            langfuse_client: Langfuse client instance
            session_id: Session identifier
            turn_number: Turn number within the conversation
            
        Returns:
            32 hexchar lowercase trace ID string
        """
        seed = f"{session_id}-turn-{turn_number}"
        trace_id = langfuse_client.create_trace_id(seed=seed)
        logger.debug(f"Generated trace ID for {seed}: {trace_id}")
        return trace_id
    
    @staticmethod
    def log_session_start(session_id: str, persona: str, scenario: str):
        """Log the start of a new session"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🆕 NEW SESSION: {session_id}")
        logger.info(f"Persona: {persona[:100]}...")
        logger.info(f"Scenario: {scenario[:100]}...")
        logger.info(f"{'='*80}\n")
    
    @staticmethod
    def log_session_complete(session_id: str, num_turns: int, evaluation_scores: Dict = None):
        """Log session completion with evaluation summary"""
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ SESSION COMPLETE: {session_id}")
        logger.info(f"Total Turns: {num_turns}")
        
        if evaluation_scores:
            logger.info(f"\nSession Evaluation Scores:")
            for metric, score in evaluation_scores.items():
                logger.info(f"  • {metric}: {score}")
        
        logger.info(f"{'='*80}\n")

def format_conversation_for_display(conversation_log: list, max_length: int = 150) -> str:
    """
    Format conversation log for display or logging
    
    Args:
        conversation_log: List of turn dictionaries
        max_length: Maximum length for each message
        
    Returns:
        Formatted conversation string
    """
    formatted = []
    for turn in conversation_log:
        turn_num = turn.get('turn', '?')
        user_msg = turn.get('user', '')[:max_length]
        assistant_msg = turn.get('assistant', '')[:max_length]
        
        formatted.append(f"Turn {turn_num}:")
        formatted.append(f"  User: {user_msg}...")
        formatted.append(f"  Assistant: {assistant_msg}...")
        formatted.append("")
    
    return "\n".join(formatted)

def validate_session_data(conversation_log: list, session_id: str) -> bool:
    """
    Validate that session data is complete and ready for evaluation
    
    Args:
        conversation_log: List of conversation turns
        session_id: Session identifier
        
    Returns:
        True if valid, False otherwise
    """
    if not conversation_log:
        logger.error(f"Empty conversation log for session {session_id}")
        return False
    
    for turn in conversation_log:
        if not turn.get('user') or not turn.get('assistant'):
            logger.error(f"Incomplete turn data in session {session_id}: {turn}")
            return False
    
    logger.debug(f"Session data validated for {session_id}: {len(conversation_log)} turns")
    return True