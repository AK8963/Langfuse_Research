"""
Evaluation metrics module for LLM-as-a-Judge evaluations
Provides trace-level evaluators only (session-level removed)
"""

from typing import Dict, List, Any
import json
from langfuse import Evaluation
from langfuse.openai import OpenAI
from utils.utils import logger, safe_json_parse


class EvaluationMetrics:
    """Handle all evaluation metrics for traces"""
    
    def __init__(self, judge_model: str, ollama_config: Dict, metrics_config: Dict):
        """
        Initialize evaluation metrics
        
        Args:
            judge_model: Model to use for LLM-as-a-Judge evaluation
            ollama_config: Ollama configuration dict with base_url and api_key
            metrics_config: Metrics configuration from config.json
        """
        self.judge_model = judge_model
        self.client = OpenAI(
            base_url=ollama_config['base_url'],
            api_key=ollama_config['api_key']
        )
        self.metrics_config = metrics_config
        
        logger.info(f"Initialized EvaluationMetrics with {len(metrics_config)} metrics")

    ######### Helper Functions

    # Combines the metrics prompts from the config
    def _generate_evaluation_prompt(self, metric_name: str, turn_data: Dict, 
                                   persona: str, scenario: str) -> str:
        """
        Generate evaluation prompt from config template (turn-level only)
        
        Args:
            metric_name: Name of the metric (relevance, accuracy, etc.)
            turn_data: Single turn data for turn-level evaluation
            persona: User persona
            scenario: User scenario
        """
        if metric_name not in self.metrics_config:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        metric_config = self.metrics_config[metric_name]
        description = metric_config['description']
        scale = metric_config['scale']
        
        # Build scale description
        scale_text = f"Score the {metric_name} ({scale['min']}-{scale['max']}):\n"
        for score, label in sorted(metric_config['scale']['labels'].items(), 
                                  key=lambda x: int(x[0]), reverse=True):
            scale_text += f"- {score}: {label}\n"
        
        # Turn-level evaluation
        return f"""You are evaluating the {metric_name.upper()} of an assistant's response.
                # USER CONTEXT
                Persona: {persona}
                Scenario: {scenario}

                # TURN {turn_data['turn']}
                User: {turn_data['user']}
                Assistant: {turn_data['assistant']}

                # EVALUATION CRITERIA
                {scale_text}

                Return ONLY a JSON object:
                {{
                "score": <int {scale['min']}-{scale['max']}>,
                "reasoning": "Brief explanation"
                }}
            """
    
    # LLM AS A JUDGE - calls judge model for evaluation
    def _evaluate_with_metric(self, metric_name: str, turn_data: Dict,
                            persona: str = "", scenario: str = "") -> Dict:
        """
        Evaluate using a specific metric
        
        Returns: Dict with score and reasoning
        """
        try:
            evaluation_prompt = self._generate_evaluation_prompt(
                metric_name=metric_name,
                turn_data=turn_data,
                persona=persona,
                scenario=scenario
            )
            
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
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to evaluate {metric_name}: {str(e)}")
            return {"score": 0, "reasoning": f"Evaluation error: {str(e)}"}
        

    # LLM As A JUDGE - evaluates every single turns.
    def evaluate_turn(self, turn_data: Dict, persona: str, scenario: str) -> Dict:
        """
        Evaluate a single turn with all metrics
        Returns: Dict with scores, overall_score, and reasoning
        """
        scores = {}
        all_reasoning = []
        
        for metric_name in self.metrics_config.keys():
            result = self._evaluate_with_metric(
                metric_name=metric_name,
                turn_data=turn_data,
                persona=persona,
                scenario=scenario
            )
            scores[metric_name] = result["score"]
            all_reasoning.append(f"{metric_name}: {result['reasoning']}")
        
        # Calculate overall score (normalized to 0-1)
        max_score = self.metrics_config[list(self.metrics_config.keys())[0]]['scale']['max']
        overall_score = sum(scores.values()) / len(scores) / max_score if scores else 0.0
        
        return {
            "scores": scores,
            "overall_score": overall_score,
            "reasoning": " | ".join(all_reasoning)
        }
    


    # Item-Level Evaluator
    def create_evaluator(self, metric_name: str):
        """
        Create evaluator that reuses pre-computed scores from conversation_log
        """
        if metric_name not in self.metrics_config:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        def evaluator(*, input, output, expected_output, metadata, **kwargs):
            try:
                if not output or not isinstance(output, dict):
                    logger.warning(f"Invalid output structure: {output}")
                    return Evaluation(
                        name=metric_name, 
                        value=0, 
                        comment="Invalid output structure"
                    )
                
                conversation_log = output.get('conversation_log', [])
                
                if not conversation_log:
                    return Evaluation(
                        name=metric_name, 
                        value=0, 
                        comment="No conversation log found"
                    )
                
                # Extract pre-computed scores instead of re-evaluating
                total_score = 0
                for turn_data in conversation_log:
                    # Check if evaluation already exists
                    if 'evaluation' in turn_data and 'scores' in turn_data['evaluation']:
                        # Reuse existing score
                        score = turn_data['evaluation']['scores'].get(metric_name, 0)
                        total_score += score
                    else:
                        # Fallback: evaluate if not pre-computed
                        logger.warning(f"Turn {turn_data.get('turn')} missing evaluation, computing now")
                        persona = input.get('persona', '') if input else ''
                        scenario = input.get('scenario', '') if input else ''
                        
                        turn_eval = {
                            'turn': turn_data.get('turn', 0),
                            'user': turn_data.get('user', ''),
                            'assistant': turn_data.get('assistant', '')
                        }
                        
                        result = self._evaluate_with_metric(
                            metric_name=metric_name,
                            turn_data=turn_eval,
                            persona=persona,
                            scenario=scenario
                        )
                        total_score += result["score"]
                
                max_score = self.metrics_config[metric_name]['scale']['max']
                avg_score = total_score / len(conversation_log)
                
                return Evaluation(
                    name=metric_name,
                    value=avg_score,
                    comment=f"Average {metric_name} across {len(conversation_log)} turns"
                )
                
            except Exception as e:
                logger.error(f"Failed to evaluate {metric_name}: {str(e)}", exc_info=True)
                return Evaluation(name=metric_name, value=0, comment=str(e))
        
        return evaluator
    
    def get_all_evaluators(self) -> List:
        """
        Get all evaluator functions for all configured metrics
        Returns: List of evaluator functions
        """
        return [self.create_evaluator(metric_name) 
                for metric_name in self.metrics_config.keys()]
    
    def create_average_quality_run_evaluator(self):
        """Create run-level evaluator for average quality across all items"""
        def run_evaluator(*, item_results, **kwargs):
            try:
                all_scores = []
                
                for result in item_results:
                    if hasattr(result, 'evaluations') and result.evaluations:
                        for eval in result.evaluations:
                            if eval.name in self.metrics_config.keys():
                                all_scores.append(eval.value)
                
                if not all_scores:
                    return Evaluation(
                        name="avg_quality", 
                        value=None, 
                        comment="No scores available"
                    )
                
                avg = sum(all_scores) / len(all_scores)
                max_score = self.metrics_config[list(self.metrics_config.keys())[0]]['scale']['max']
                
                return Evaluation(
                    name="avg_quality",
                    value=avg,
                    comment=f"Average quality across all metrics and items: {avg:.2f}/{max_score}"
                )
            
            except Exception as e:
                logger.error(f"Failed to calculate average quality: {str(e)}", exc_info=True)
                return Evaluation(name="avg_quality", value=None, comment=str(e))
        
        return run_evaluator