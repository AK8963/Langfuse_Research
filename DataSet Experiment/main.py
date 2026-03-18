"""Main execution script for multi-turn conversation experiments"""

from langfuse import get_client, propagate_attributes
from agents.chatbot import LlamaChatbot
from agents.simulated_user import SimulatedUser
from metrics.metric import EvaluationMetrics
from utils.utils import (
    load_config, 
    validate_langfuse_connection, 
    log_turn_info, 
    SessionManager,
    logger
)
from typing import Dict, List

# Load configuration
config = load_config()

# Initialize Langfuse client
langfuse = get_client()
if not validate_langfuse_connection(langfuse):
    raise RuntimeError("❌ Langfuse authentication failed")

# Initialize evaluation metrics with config
evaluator = EvaluationMetrics(
    judge_model=config['models']['judge_model'],
    ollama_config=config['ollama'],
    metrics_config=config['evaluation_metrics']
)

def simulate_single_turn(
    chatbot: LlamaChatbot, 
    simulated_user: SimulatedUser, 
    turn_number: int, 
    session_id: str,
    persona: str,
    scenario: str,
    is_first: bool = False
) -> Dict:
    """Simulate a single turn as a separate trace within the session"""
    trace_id = SessionManager.generate_trace_id(langfuse, session_id, turn_number)
    
    with langfuse.start_as_current_observation(
        as_type="span",
        name=f"Turn {turn_number}",
        trace_context={"trace_id": trace_id}
    ) as span:
        with propagate_attributes(session_id=session_id):
            # User generates message
            user_message = simulated_user.generate_message(is_first_turn=is_first)
            
            # Assistant responds
            assistant_response = chatbot.chat(user_message, turn_number)
            
            # Update simulated user's memory
            simulated_user.update_history(user_message, assistant_response)
            
            # Update trace with turn information
            span.update_trace(
                name=f"Turn {turn_number}",
                input={"user_message": user_message},
                output={"assistant_response": assistant_response},
                metadata={
                    "turn_number": turn_number,
                    "persona": persona,
                    "scenario": scenario
                }
            )
            
            return {
                "turn": turn_number,
                "user": user_message,
                "assistant": assistant_response,
                "trace_id": trace_id
            }

def simulate_continuous_conversation(
    persona: str, 
    scenario: str, 
    session_id: str, 
    max_turns: int = 6
) -> Dict:
    """Simulate a continuous conversation with each turn as a separate trace"""
    chatbot = LlamaChatbot(
        model=config['models']['answer_model'],
        ollama_config=config['ollama'],
        session_id=session_id
    )
    
    simulated_user = SimulatedUser(
        persona=persona,
        scenario=scenario,
        model=config['models']['answer_model_alt'],
        ollama_config=config['ollama']
    )
    
    conversation_log = []
    
    logger.info(f"\n{'='*80}")
    logger.info(f"SESSION: {session_id}")
    logger.info(f"SCENARIO: {scenario}")
    logger.info(f"{'='*80}\n")
    
    for turn in range(1, max_turns + 1):
        is_first = (turn == 1)
        turn_result = simulate_single_turn(
            chatbot, 
            simulated_user, 
            turn, 
            session_id,
            persona,
            scenario,
            is_first
        )
        
        conversation_log.append(turn_result)
        log_turn_info(turn, turn_result['user'], turn_result['assistant'])
    
    return {
        "conversation_log": conversation_log,
        "session_id": session_id,
        "num_turns": len(conversation_log)
    }

def evaluate_and_score_traces(
    conversation_log: List[Dict],
    persona: str,
    scenario: str
):
    """Evaluate each turn and create individual trace scores"""
    logger.info(f"⚖️ Evaluating each turn...")
    evaluations = []
    
    for turn_data in conversation_log:
        # Use unified evaluate_turn method
        evaluation = evaluator.evaluate_turn(turn_data, persona, scenario)
        evaluations.append(evaluation)
        trace_id = turn_data["trace_id"]
        
        # Create individual metric scores for this trace
        for metric, value in evaluation["scores"].items():
            langfuse.create_score(
                trace_id=trace_id,
                name=metric,
                value=value,
                data_type="NUMERIC",
                comment=evaluation["reasoning"]
            )
        
        # Create overall trace score
        langfuse.create_score(
            trace_id=trace_id,
            name="overall_quality",
            value=evaluation["overall_score"],
            data_type="NUMERIC",
            comment=evaluation["reasoning"]
        )
        
        logger.info(f"   Turn {turn_data['turn']} - Score: {evaluation['overall_score']:.2f}")
    
    return evaluations





def run_task(*, item, **kwargs) -> Dict:
    """
    Task function for Langfuse experiment
    This function is called by dataset.run_experiment() for each dataset item
    
    IMPORTANT: The return value is passed to evaluators as 'output'
    """
    persona = item.input.get("persona", "")
    scenario = item.input.get("scenario", "")
    
    session_id = SessionManager.generate_session_id(item.id, prefix="session8")
    
    logger.info(f"\n🔄 Starting continuous multi-turn conversation...")
    
    # Simulate conversation
    result = simulate_continuous_conversation(
        persona=persona,
        scenario=scenario,
        session_id=session_id,
        max_turns=config['dataset']['max_turns']
    )
    
    logger.info(f"✅ Completed {result['num_turns']} turns under session: {session_id}")
    
    # Evaluate and score each trace separately
    trace_evaluations = evaluate_and_score_traces(
        result["conversation_log"],
        persona,
        scenario
    )
    
    logger.info(f"{'='*80}\n")
    
    # Return structure that evaluators will receive as 'output'
    return {
        "session_id": session_id,
        "num_turns": result["num_turns"],
        "conversation_log": result["conversation_log"],
        "trace_evaluations": trace_evaluations
    }





if __name__ == "__main__":
    dataset = langfuse.get_dataset(config['dataset']['name'])
    
    logger.info(f"📚 Fetching dataset '{config['dataset']['name']}' from Langfuse...")
    logger.info(f"Found {len(dataset.items)} items.\n")
    
    # Get all evaluators from config
    evaluators = evaluator.get_all_evaluators()
    run_evaluators = [ evaluator.create_average_quality_run_evaluator() ]
    logger.info("🚀 Starting experiment with evaluators...")

    result = dataset.run_experiment(
        name="unified-metrics-template-v1",
        description="Using unified metrics template from config with automatic evaluation",
        task=run_task,
        evaluators=evaluators,
        run_evaluators=run_evaluators
    )
    
    langfuse.flush()
    
    logger.info("\n" + "="*80)
    logger.info("✅ EXPERIMENT COMPLETE!")
    logger.info("="*80)
    logger.info(f"✓ Each turn is stored as a separate trace")
    logger.info(f"✓ All turns grouped under their session IDs")
    logger.info(f"✓ Each trace has individual evaluation scores (manual)")
    logger.info(f"✓ Evaluators automatically scored each dataset item")
    logger.info(f"✓ Run evaluators calculated aggregate metrics")
    logger.info(f"✓ All metrics driven by config templates")
    logger.info(f"✓ View results in Langfuse UI under Dataset Runs")
    logger.info("="*80)