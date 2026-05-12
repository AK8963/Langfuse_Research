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
        
        # Resolve path relative to project root
        if not os.path.isabs(config_path):
            project_root = Path(__file__).parent.parent
            config_path = project_root / config_path
        
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

def handle_error(error: Exception, context: str = "") -> str:
    """Centralized error handling"""
    error_msg = f"Error in {context}: {str(error)}"
    logger.error(error_msg, exc_info=True)
    return error_msg

class SessionManager:
    """Manage session IDs and conversation state"""
    
    @staticmethod
    def generate_session_id(item_id: str, prefix: str = "session") -> str:
        """Generate unique session ID"""
        return f"{prefix}-{item_id}"
    
    @staticmethod
    def generate_trace_id(langfuse_client, session_id: str, turn_number: int) -> str:
        """Generate unique trace ID for a turn"""
        return langfuse_client.create_trace_id(seed=f"{session_id}-turn-{turn_number}")