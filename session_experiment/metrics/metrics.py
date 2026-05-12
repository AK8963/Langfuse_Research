"""
Evaluation metrics module for LLM-as-a-Judge evaluations
Provides session-level evaluators only
"""

from typing import Dict, List, Any
import json
from langfuse import Evaluation
from langfuse.openai import OpenAI
from utils.utils import logger, safe_json_parse, handle_error


class EvaluationMetrics:
    """Handle session-level evaluation metrics only"""
    
    def __init__(self, judge_model: str, ollama_config: Dict):
        self.judge_model = judge_model
        self.client = OpenAI(
            base_url=ollama_config['base_url'],
            api_key=ollama_config['api_key']
        )
    
    # ========================================================================
    # SESSION-LEVEL EVALUATION PROMPTS
    # ========================================================================

    def _get_session_relevance_prompt(self, conversation_log: List[Dict], persona: str, scenario: str) -> str:
        """Evaluate overall session relevance across all turns"""
        conversation_text = "\n".join([
            f"Turn {t['turn']}:\nUser: {t['user']}\nAssistant: {t['assistant']}\n"
            for t in conversation_log
        ])
        
        return f"""You are evaluating the overall RELEVANCE of a multi-turn conversation session.

# USER CONTEXT
Persona: {persona}
Scenario: {scenario}

# FULL CONVERSATION
{conversation_text}

# EVALUATION CRITERIA
Score the overall session relevance (1-5):
- 5: All responses stay on topic throughout
- 4: Mostly relevant with minor deviations
- 3: Partially relevant
- 2: Frequently off-topic
- 1: Consistently irrelevant

Return ONLY a JSON object:
{{
  "score": <int 1-5>,
  "reasoning": "Brief explanation of session-level relevance"
}}
"""

    def _get_session_accuracy_prompt(self, conversation_log: List[Dict], persona: str, scenario: str) -> str:
        """Evaluate overall session accuracy across all turns"""
        conversation_text = "\n".join([
            f"Turn {t['turn']}:\nUser: {t['user']}\nAssistant: {t['assistant']}\n"
            for t in conversation_log
        ])
        
        return f"""You are evaluating the overall ACCURACY of a multi-turn conversation session.

# USER CONTEXT
Persona: {persona}
Scenario: {scenario}

# FULL CONVERSATION
{conversation_text}

# EVALUATION CRITERIA
Score the overall session accuracy (1-5):
- 5: All information consistently accurate
- 4: Mostly accurate throughout
- 3: Mixed accuracy
- 2: Frequently inaccurate
- 1: Consistently inaccurate

Return ONLY a JSON object:
{{
  "score": <int 1-5>,
  "reasoning": "Brief explanation of session-level accuracy"
}}
"""

    def _get_session_helpfulness_prompt(self, conversation_log: List[Dict], persona: str, scenario: str) -> str:
        """Evaluate overall session helpfulness across all turns"""
        conversation_text = "\n".join([
            f"Turn {t['turn']}:\nUser: {t['user']}\nAssistant: {t['assistant']}\n"
            for t in conversation_log
        ])
        
        return f"""You are evaluating the overall HELPFULNESS of a multi-turn conversation session.

# USER CONTEXT
Persona: {persona}
Scenario: {scenario}

# FULL CONVERSATION
{conversation_text}

# EVALUATION CRITERIA
Score the overall session helpfulness (1-5):
- 5: Consistently helpful throughout the conversation
- 4: Generally helpful with good guidance
- 3: Somewhat helpful
- 2: Minimally helpful
- 1: Not helpful overall

Return ONLY a JSON object:
{{
  "score": <int 1-5>,
  "reasoning": "Brief explanation of session-level helpfulness"
}}
"""

    def _get_session_clarity_prompt(self, conversation_log: List[Dict], persona: str, scenario: str) -> str:
        """Evaluate overall session clarity across all turns"""
        conversation_text = "\n".join([
            f"Turn {t['turn']}:\nUser: {t['user']}\nAssistant: {t['assistant']}\n"
            for t in conversation_log
        ])
        
        return f"""You are evaluating the overall CLARITY of a multi-turn conversation session.

# USER CONTEXT
Persona: {persona}
Scenario: {scenario}

# FULL CONVERSATION
{conversation_text}

# EVALUATION CRITERIA
Score the overall session clarity (1-5):
- 5: Consistently clear and well-structured throughout
- 4: Generally clear communication
- 3: Moderately clear
- 2: Often unclear or confusing
- 1: Consistently unclear

Return ONLY a JSON object:
{{
  "score": <int 1-5>,
  "reasoning": "Brief explanation of session-level clarity"
}}
"""

  
    
    def evaluate_session(self, conversation_log: List[Dict], persona: str, scenario: str) -> Dict:
        """
        Evaluate entire session with session-level metrics
        Returns: Dict with session_scores, overall_session_score, and reasoning
        """
        session_metrics = {
            "session_relevance": self._get_session_relevance_prompt,
            "session_accuracy": self._get_session_accuracy_prompt,
            "session_helpfulness": self._get_session_helpfulness_prompt,
            "session_clarity": self._get_session_clarity_prompt
        }
        
        session_scores = {}
        all_reasoning = []
        
        for metric_name, prompt_func in session_metrics.items():
            try:
                evaluation_prompt = prompt_func(conversation_log, persona, scenario)
                
                response = self.client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": "You are an expert conversation evaluator."},
                        {"role": "user", "content": evaluation_prompt}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                
                result = safe_json_parse(
                    response.choices[0].message.content,
                    {"score": 0, "reasoning": "Parse error"}
                )
                session_scores[metric_name] = result["score"]
                all_reasoning.append(f"{metric_name}: {result['reasoning']}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate session {metric_name}: {str(e)}")
                session_scores[metric_name] = 0
        
        overall_session_score = sum(session_scores.values()) / len(session_scores) / 5.0 if session_scores else 0.0
        
        return {
            "session_scores": session_scores,
            "overall_session_score": overall_session_score,
            "reasoning": " | ".join(all_reasoning)
        }

    # ========================================================================
    # SESSION-LEVEL EVALUATOR FUNCTIONS (for dataset.run_experiment)
    # These receive the OUTPUT of the task function
    # ========================================================================

    # def create_session_relevance_evaluator(self):
    #     """Create session-level evaluator function for relevance"""
    #     def evaluator(*, input, output, expected_output, metadata, **kwargs):
    #         try:
    #             if not output or not isinstance(output, dict):
    #                 logger.warning(f"Invalid output structure: {output}")
    #                 return Evaluation(name="session_relevance", value=0, comment="Invalid output structure")
                
    #             conversation_log = output.get('conversation_log', [])
    #             persona = input.get('persona', '') if input else ''
    #             scenario = input.get('scenario', '') if input else ''
                
    #             if not conversation_log:
    #                 return Evaluation(name="session_relevance", value=0, comment="No conversation log found")
                
    #             evaluation_prompt = self._get_session_relevance_prompt(conversation_log, persona, scenario)
                
    #             response = self.client.chat.completions.create(
    #                 model=self.judge_model,
    #                 messages=[
    #                     {"role": "system", "content": "You are an expert conversation evaluator."},
    #                     {"role": "user", "content": evaluation_prompt}
    #                 ],
    #                 temperature=0,
    #                 response_format={"type": "json_object"}
    #             )
                
    #             result = safe_json_parse(
    #                 response.choices[0].message.content,
    #                 {"score": 0, "reasoning": "Parse error"}
    #             )
                
    #             return Evaluation(
    #                 name="session_relevance",
    #                 value=result["score"],
    #                 comment=result["reasoning"]
    #             )
                
    #         except Exception as e:
    #             logger.error(f"Failed to evaluate session relevance: {str(e)}", exc_info=True)
    #             return Evaluation(name="session_relevance", value=0, comment=str(e))
        
    #     return evaluator
    
    # def create_session_accuracy_evaluator(self):
    #     """Create session-level evaluator function for accuracy"""
    #     def evaluator(*, input, output, expected_output, metadata, **kwargs):
    #         try:
    #             if not output or not isinstance(output, dict):
    #                 logger.warning(f"Invalid output structure: {output}")
    #                 return Evaluation(name="session_accuracy", value=0, comment="Invalid output structure")
                
    #             conversation_log = output.get('conversation_log', [])
    #             persona = input.get('persona', '') if input else ''
    #             scenario = input.get('scenario', '') if input else ''
                
    #             if not conversation_log:
    #                 return Evaluation(name="session_accuracy", value=0, comment="No conversation log found")
                
    #             evaluation_prompt = self._get_session_accuracy_prompt(conversation_log, persona, scenario)
                
    #             response = self.client.chat.completions.create(
    #                 model=self.judge_model,
    #                 messages=[
    #                     {"role": "system", "content": "You are an expert conversation evaluator."},
    #                     {"role": "user", "content": evaluation_prompt}
    #                 ],
    #                 temperature=0,
    #                 response_format={"type": "json_object"}
    #             )
                
    #             result = safe_json_parse(
    #                 response.choices[0].message.content,
    #                 {"score": 0, "reasoning": "Parse error"}
    #             )
                
    #             return Evaluation(
    #                 name="session_accuracy",
    #                 value=result["score"],
    #                 comment=result["reasoning"]
    #             )
                
    #         except Exception as e:
    #             logger.error(f"Failed to evaluate session accuracy: {str(e)}", exc_info=True)
    #             return Evaluation(name="session_accuracy", value=0, comment=str(e))
        
    #     return evaluator
    
    # def create_session_helpfulness_evaluator(self):
    #     """Create session-level evaluator function for helpfulness"""
    #     def evaluator(*, input, output, expected_output, metadata, **kwargs):
    #         try:
    #             if not output or not isinstance(output, dict):
    #                 logger.warning(f"Invalid output structure: {output}")
    #                 return Evaluation(name="session_helpfulness", value=0, comment="Invalid output structure")
                
    #             conversation_log = output.get('conversation_log', [])
    #             persona = input.get('persona', '') if input else ''
    #             scenario = input.get('scenario', '') if input else ''
                
    #             if not conversation_log:
    #                 return Evaluation(name="session_helpfulness", value=0, comment="No conversation log found")
                
    #             evaluation_prompt = self._get_session_helpfulness_prompt(conversation_log, persona, scenario)
                
    #             response = self.client.chat.completions.create(
    #                 model=self.judge_model,
    #                 messages=[
    #                     {"role": "system", "content": "You are an expert conversation evaluator."},
    #                     {"role": "user", "content": evaluation_prompt}
    #                 ],
    #                 temperature=0,
    #                 response_format={"type": "json_object"}
    #             )
                
    #             result = safe_json_parse(
    #                 response.choices[0].message.content,
    #                 {"score": 0, "reasoning": "Parse error"}
    #             )
                
    #             return Evaluation(
    #                 name="session_helpfulness",
    #                 value=result["score"],
    #                 comment=result["reasoning"]
    #             )
                
    #         except Exception as e:
    #             logger.error(f"Failed to evaluate session helpfulness: {str(e)}", exc_info=True)
    #             return Evaluation(name="session_helpfulness", value=0, comment=str(e))
        
    #     return evaluator
    
    # def create_session_clarity_evaluator(self):
    #     """Create session-level evaluator function for clarity"""
    #     def evaluator(*, input, output, expected_output, metadata, **kwargs):
    #         try:
    #             if not output or not isinstance(output, dict):
    #                 logger.warning(f"Invalid output structure: {output}")
    #                 return Evaluation(name="session_clarity", value=0, comment="Invalid output structure")
                
    #             conversation_log = output.get('conversation_log', [])
    #             persona = input.get('persona', '') if input else ''
    #             scenario = input.get('scenario', '') if input else ''
                
    #             if not conversation_log:
    #                 return Evaluation(name="session_clarity", value=0, comment="No conversation log found")
                
    #             evaluation_prompt = self._get_session_clarity_prompt(conversation_log, persona, scenario)
                
    #             response = self.client.chat.completions.create(
    #                 model=self.judge_model,
    #                 messages=[
    #                     {"role": "system", "content": "You are an expert conversation evaluator."},
    #                     {"role": "user", "content": evaluation_prompt}
    #                 ],
    #                 temperature=0,
    #                 response_format={"type": "json_object"}
    #             )
                
    #             result = safe_json_parse(
    #                 response.choices[0].message.content,
    #                 {"score": 0, "reasoning": "Parse error"}
    #             )
                
    #             return Evaluation(
    #                 name="session_clarity",
    #                 value=result["score"],
    #                 comment=result["reasoning"]
    #             )
                
    #         except Exception as e:
    #             logger.error(f"Failed to evaluate session clarity: {str(e)}", exc_info=True)
    #             return Evaluation(name="session_clarity", value=0, comment=str(e))
        
    #     return evaluator
    
    # ========================================================================
    # RUN-LEVEL EVALUATOR (aggregate metrics across all dataset items)
    # ========================================================================
    
    def create_session_quality_run_evaluator(self):
        """Create run-level evaluator for overall session quality"""
        def run_evaluator(*, item_results, **kwargs):
            """Calculate overall session quality from all items"""
            try:
                all_session_scores = []
                
                for result in item_results:
                    if hasattr(result, 'evaluations') and result.evaluations:
                        for eval in result.evaluations:
                            if eval.name in ["session_relevance", "session_accuracy", 
                                           "session_helpfulness", "session_clarity"]:
                                all_session_scores.append(eval.value)
                
                if not all_session_scores:
                    return Evaluation(
                        name="avg_session_quality",
                        value=None,
                        comment="No session scores available"
                    )
                
                avg = sum(all_session_scores) / len(all_session_scores)
                
                return Evaluation(
                    name="avg_session_quality",
                    value=avg,
                    comment=f"Average session quality across all metrics and items: {avg:.2f}/5"
                )
            
            except Exception as e:
                logger.error(f"Failed to calculate session quality: {str(e)}", exc_info=True)
                return Evaluation(name="avg_session_quality", value=None, comment=str)