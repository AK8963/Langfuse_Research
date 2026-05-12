"""
Utility functions for session management, configuration loading, and logging
"""

import json
import os
import logging
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


# ============================================================================
# LOGGING SETUP
# ============================================================================

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger


logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_config(config_path: str = "config/config.json") -> Dict:  # Changed default
    """
    Load configuration from JSON file with environment variable substitution
    Searches in multiple locations: current dir, script dir, parent dirs
    
    Args:
        config_path: Path to configuration file (relative or absolute)
        
    Returns:
        Configuration dictionary
    """
    try:
        # Try multiple possible locations
        search_paths = [
            Path(config_path),  # As provided
            Path(__file__).parent.parent / config_path,  # Parent of utils directory
            Path(__file__).parent / config_path,  # Same directory as utils
            Path.cwd() / config_path,  # Explicit current working directory
            Path.cwd() / "config" / "config.json",  # Explicit config folder
        ]
        
        config_file = None
        for path in search_paths:
            if path.exists():
                config_file = path
                logger.info(f"Found config at: {config_file.absolute()}")
                break
        
        if config_file is None:
            # Print debug info
            logger.error(f"Config file '{config_path}' not found in any of these locations:")
            for path in search_paths:
                logger.error(f"  - {path.absolute()} (exists: {path.exists()})")
            logger.error(f"Current working directory: {Path.cwd()}")
            logger.error(f"Script location: {Path(__file__).parent}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Substitute environment variables
        config = _substitute_env_vars(config)
        
        # Validate required fields
        _validate_config(config)
        
        logger.info(f"✓ Configuration loaded from {config_file.absolute()}")
        return config
    
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def _substitute_env_vars(config: Any) -> Any:
    """
    Recursively substitute environment variables in configuration
    Format: ${ENV_VAR_NAME}
    """
    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        value = os.getenv(env_var)
        if value is None:
            logger.warning(f"Environment variable {env_var} not set, using placeholder")
            return config
        return value
    return config


def _validate_config(config: Dict) -> None:
    """Validate required configuration fields"""
    required_sections = ['ollama', 'langfuse', 'models', 'dataset', 'evaluators']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate evaluators
    if 'session_metrics' not in config['evaluators']:
        raise ValueError("No session_metrics defined in evaluators configuration")
    
    if not config['evaluators']['session_metrics']:
        raise ValueError("session_metrics list is empty")


# ============================================================================
# LANGFUSE CONNECTION VALIDATION
# ============================================================================

def validate_langfuse_connection(langfuse_client) -> bool:
    """
    Validate Langfuse client connection
    
    Args:
        langfuse_client: Langfuse client instance
        
    Returns:
        True if connection is valid, False otherwise
    """
    try:
        # Try to authenticate
        langfuse_client.auth_check()
        logger.info("✓ Langfuse connection validated")
        return True
    except Exception as e:
        logger.error(f"Langfuse connection validation failed: {e}")
        return False


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class SessionManager:
    """Utility class for managing conversation sessions"""
    
    @staticmethod
    def generate_session_id(item_id: str, prefix: str = "session") -> str:
        """
        Generate a unique session ID
        
        Args:
            item_id: Dataset item ID
            prefix: Prefix for session ID
            
        Returns:
            Unique session ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(item_id.encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{hash_suffix}"
    
    @staticmethod
    def generate_trace_id(langfuse_client, session_id: str, turn_number: int) -> str:
        """
        Generate a deterministic trace ID for a turn
        
        Args:
            langfuse_client: Langfuse client instance
            session_id: Session identifier
            turn_number: Turn number
            
        Returns:
            Deterministic trace ID
        """
        trace_base = f"{session_id}_turn_{turn_number}"
        return hashlib.md5(trace_base.encode()).hexdigest()
    
    @staticmethod
    def log_session_start(session_id: str, persona: str, scenario: str) -> None:
        """
        Log session start information
        
        Args:
            session_id: Session identifier
            persona: User persona
            scenario: Conversation scenario
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🎬 SESSION START: {session_id}")
        logger.info(f"Persona: {persona[:100]}...")
        logger.info(f"Scenario: {scenario[:100]}...")
        logger.info(f"{'='*80}\n")
    
    @staticmethod
    def log_session_complete(session_id: str, num_turns: int, metrics: Dict) -> None:
        """
        Log session completion information
        
        Args:
            session_id: Session identifier
            num_turns: Number of turns completed
            metrics: Session metrics (if available)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ SESSION COMPLETE: {session_id}")
        logger.info(f"Turns completed: {num_turns}")
        
        if metrics:
            logger.info("Session metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
        
        logger.info(f"{'='*80}\n")


# ============================================================================
# JSON UTILITIES
# ============================================================================

def safe_json_parse(json_string: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback
    
    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        return default if default is not None else {}


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely serialize object to JSON string
    
    Args:
        obj: Object to serialize
        default: Default string if serialization fails
        
    Returns:
        JSON string or default
    """
    try:
        return json.dumps(obj, indent=2)
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON serialization error: {e}")
        return default


# ============================================================================
# ERROR HANDLING
# ============================================================================

def handle_error(error: Exception, context: str = "") -> Dict:
    """
    Handle and log errors with context
    
    Args:
        error: Exception that occurred
        context: Context information
        
    Returns:
        Error dictionary
    """
    error_msg = f"{context}: {str(error)}" if context else str(error)
    logger.error(error_msg, exc_info=True)
    
    return {
        "error": True,
        "message": error_msg,
        "type": type(error).__name__
    }


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class Timer:
    """Simple context manager for timing operations"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"⏱️  {self.name} started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"⏱️  {self.name} completed in {duration:.2f}s")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0