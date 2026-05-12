"""
Scalable session-level LLM-as-a-Judge evaluation module
Supports JSON-based configuration for all metrics
"""

from typing import Dict, List, Any, Optional
import json
import time
from langfuse import Evaluation
from langfuse.openai import OpenAI


class SessionEvaluator:
    """Scalable session-level evaluator with JSON configuration"""
    
    def __init__(self, judge_model: str, ollama_config: Dict, evaluator_config: Dict, logger):
        """
        Initialize evaluator with configuration
        
        Args:
            judge_model: Model to use for evaluation
            ollama_config: Ollama connection configuration
            evaluator_config: Evaluator configuration from config.json
            logger: Logger instance
        """
        self.judge_model = judge_model
        self.logger = logger
        self.client = OpenAI(
            base_url=ollama_config['base_url'],
            api_key=ollama_config['api_key']
        )
        
        # Load evaluator configurations
        self.session_metrics = [
            metric for metric in evaluator_config.get('session_metrics', [])
            if metric.get('enabled', True)
        ]
        
        # Evaluation settings
        self.max_retries = evaluator_config.get('retry_attempts', 3)
        self.timeout = evaluator_config.get('timeout_seconds', 30)
        
        self.logger.info(f"✓ Initialized {len(self.session_metrics)} session evaluators")
    
    def _format_conversation(self, conversation_log: List[Dict]) -> str:
        """Format conversation log into readable text"""
        return "\n".join([
            f"Turn {turn['turn']}:\nUser: {turn['user']}\nAssistant: {turn['assistant']}\n"
            for turn in conversation_log
        ])
    
    def _call_llm_judge(self, system_prompt: str, evaluation_prompt: str, metric_name: str) -> Dict:
        """
        Call LLM judge with retry logic and error handling
        
        Args:
            system_prompt: System prompt for the judge
            evaluation_prompt: Formatted evaluation prompt
            metric_name: Name of the metric being evaluated
            
        Returns:
            Dict with score and reasoning
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": evaluation_prompt}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=self.timeout
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Validate result
                if 'score' not in result or 'reasoning' not in result:
                    raise ValueError(f"Invalid result structure: {result}")
                
                return result
                
            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"[{metric_name}] JSON parse error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    return {"score": 0, "reasoning": f"JSON parse error: {str(e)}"}
                time.sleep(2 ** attempt)
                
            except Exception as e:
                self.logger.error(
                    f"[{metric_name}] Evaluation error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    return {"score": 0, "reasoning": f"Evaluation error: {str(e)}"}
                time.sleep(2 ** attempt)
        
        return {"score": 0, "reasoning": "Max retries exceeded"}
    
    def evaluate_single_metric(self, 
                              metric_config: Dict,
                              conversation_log: List[Dict],
                              persona: str,
                              scenario: str) -> Dict:
        """
        Evaluate a single metric for a session
        
        Args:
            metric_config: Configuration for the metric
            conversation_log: Full conversation history
            persona: User persona
            scenario: Conversation scenario
            
        Returns:
            Dict with score, reasoning, and metadata
        """
        try:
            conversation_text = self._format_conversation(conversation_log)
            
            # Format the evaluation prompt
            evaluation_prompt = metric_config['evaluation_prompt'].format(
                persona=persona,
                scenario=scenario,
                conversation_text=conversation_text
            )
            
            # Call LLM judge
            result = self._call_llm_judge(
                system_prompt=metric_config.get('system_prompt', 
                                               "You are an expert evaluator."),
                evaluation_prompt=evaluation_prompt,
                metric_name=metric_config['name']
            )
            
            return {
                'name': metric_config['name'],
                'score': result['score'],
                'reasoning': result['reasoning'],
                'weight': metric_config.get('weight', 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate {metric_config['name']}: {e}")
            return {
                'name': metric_config['name'],
                'score': 0,
                'reasoning': f"Evaluation failed: {str(e)}",
                'weight': metric_config.get('weight', 1.0)
            }
    
    def evaluate_session(self, 
                        conversation_log: List[Dict],
                        persona: str,
                        scenario: str) -> Dict:
        """
        Evaluate entire session with all configured metrics
        
        Args:
            conversation_log: Full conversation history
            persona: User persona
            scenario: Conversation scenario
            
        Returns:
            Dict with all scores and aggregate metrics
        """
        results = []
        
        for metric_config in self.session_metrics:
            result = self.evaluate_single_metric(
                metric_config, conversation_log, persona, scenario
            )
            results.append(result)
            self.logger.info(
                f"✓ {result['name']}: {result['score']}/5 - {result['reasoning'][:50]}..."
            )
        
        # Calculate weighted average
        total_weight = sum(r['weight'] for r in results)
        weighted_score = sum(r['score'] * r['weight'] for r in results) / total_weight if total_weight > 0 else 0
        
        # Calculate normalized score (0-1 range)
        normalized_score = weighted_score / 5.0
        
        return {
            'individual_scores': results,
            'weighted_average': weighted_score,
            'normalized_score': normalized_score,
            'num_metrics': len(results)
        }
    
    def create_evaluator_function(self, metric_name: str):
        """
        Create an evaluator function for a specific metric
        Compatible with dataset.run_experiment()
        
        Args:
            metric_name: Name of the metric to evaluate
            
        Returns:
            Evaluator function
        """
        # Find metric configuration
        metric_config = next(
            (m for m in self.session_metrics if m['name'] == metric_name),
            None
        )
        
        if not metric_config:
            raise ValueError(f"Metric '{metric_name}' not found in configuration")
        
        def evaluator(*, input, output, expected_output, metadata, **kwargs):
            """Evaluator function for Langfuse experiments"""
            try:
                # Validate output
                if not output or not isinstance(output, dict):
                    self.logger.warning(f"[{metric_name}] Invalid output structure")
                    return Evaluation(
                        name=metric_name,
                        value=0,
                        comment="Invalid output structure"
                    )
                
                # Extract data
                conversation_log = output.get('conversation_log', [])
                persona = input.get('persona', '') if input else ''
                scenario = input.get('scenario', '') if input else ''
                
                if not conversation_log:
                    return Evaluation(
                        name=metric_name,
                        value=0,
                        comment="No conversation log found"
                    )
                
                # Evaluate
                result = self.evaluate_single_metric(
                    metric_config, conversation_log, persona, scenario
                )
                
                return Evaluation(
                    name=metric_name,
                    value=result['score'],
                    comment=result['reasoning']
                )
                
            except Exception as e:
                self.logger.error(f"[{metric_name}] Evaluator function error: {e}")
                return Evaluation(
                    name=metric_name,
                    value=0,
                    comment=f"Error: {str(e)}"
                )
        
        return evaluator
    
    def create_all_evaluators(self) -> List:
        """
        Create evaluator functions for all enabled metrics
        
        Returns:
            List of evaluator functions for dataset.run_experiment()
        """
        return [
            self.create_evaluator_function(metric['name'])
            for metric in self.session_metrics
        ]
    
    def create_run_evaluator(self):
        """
        Create run-level evaluator for aggregate metrics
        
        Returns:
            Run evaluator function
        """
        def run_evaluator(*, item_results, **kwargs):
            """Calculate aggregate metrics across all dataset items"""
            try:
                all_scores = []
                metric_names = [m['name'] for m in self.session_metrics]
                weights = {m['name']: m.get('weight', 1.0) for m in self.session_metrics}
                
                # Collect scores from all items
                for result in item_results:
                    if hasattr(result, 'evaluations') and result.evaluations:
                        for eval in result.evaluations:
                            if eval.name in metric_names:
                                all_scores.append({
                                    'name': eval.name,
                                    'value': eval.value,
                                    'weight': weights.get(eval.name, 1.0)
                                })
                
                if not all_scores:
                    return Evaluation(
                        name="weighted_session_quality",
                        value=None,
                        comment="No scores available"
                    )
                
                # Calculate weighted average
                total_weight = sum(s['weight'] for s in all_scores)
                weighted_avg = sum(s['value'] * s['weight'] for s in all_scores) / total_weight
                
                # Normalize to 0-1 range
                normalized = weighted_avg / 5.0
                
                return Evaluation(
                    name="weighted_session_quality",
                    value=normalized,
                    comment=f"Weighted average: {weighted_avg:.2f}/5 across {len(all_scores)} evaluations"
                )
                
            except Exception as e:
                self.logger.error(f"Run evaluator error: {e}")
                return Evaluation(
                    name="weighted_session_quality",
                    value=None,
                    comment=f"Error: {str(e)}"
                )
        
        return run_evaluator