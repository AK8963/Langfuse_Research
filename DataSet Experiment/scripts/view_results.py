"""Script to view experiment results from Langfuse"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langfuse import get_client
from utils.utils import load_config, validate_langfuse_connection, logger

def main():
    config = load_config()
    langfuse = get_client()
    
    if not validate_langfuse_connection(langfuse):
        raise RuntimeError("❌ Langfuse authentication failed")
    
    # Fetch recent sessions
    logger.info("Fetching recent sessions...")
    
    # Get session ID from user
    session_id = input("Enter session ID (or press Enter to skip): ")
    
    if session_id:
        try:
            # Fetch traces for this session
            traces = langfuse.api.trace.list(session_id=session_id, limit=100)
            
            logger.info(f"\n📊 Scores for session {session_id}:")
            
            # Iterate through traces and fetch scores
            for trace in traces.data:
                trace_detail = langfuse.api.trace.get(trace.id)
                
                if hasattr(trace_detail, 'scores') and trace_detail.scores:
                    logger.info(f"\nTrace ID: {trace.id}")
                    for score in trace_detail.scores:
                        logger.info(f"  - {score.name}: {score.value}")
                        if hasattr(score, 'comment') and score.comment:
                            logger.info(f"    Comment: {score.comment}")
            
            logger.info(f"\n✅ Found {len(traces.data)} traces in session")
            
        except Exception as e:
            logger.error(f"Error fetching scores: {e}")
    
    logger.info("\n✅ View the full results in Langfuse UI")

if __name__ == "__main__":
    main()