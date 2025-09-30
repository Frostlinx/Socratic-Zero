"""
Math Problem Evaluator

Main evaluator class for math problem solving assessment
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from .metrics import mean_at_k, calculate_accuracy, extract_final_answer

logger = logging.getLogger(__name__)


class MathEvaluator:
    """
    Math problem evaluator with Mean@K and accuracy metrics
    """
    
    def __init__(self):
        self.results = []
        
    def load_predictions(self, predictions_file: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load predictions from file
        
        Args:
            predictions_file: Path to predictions JSON file
            
        Returns:
            List of prediction dictionaries
        """
        predictions_file = Path(predictions_file)
        
        if not predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        logger.info(f"Loaded {len(predictions)} predictions from {predictions_file}")
        return predictions
    
    def load_ground_truth(self, ground_truth_file: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load ground truth from file
        
        Args:
            ground_truth_file: Path to ground truth JSON file
            
        Returns:
            List of ground truth dictionaries
        """
        ground_truth_file = Path(ground_truth_file)
        
        if not ground_truth_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")
        
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        logger.info(f"Loaded {len(ground_truth)} ground truth items from {ground_truth_file}")
        return ground_truth
    
    def evaluate_trajectories(self, trajectories: List[Dict[str, Any]], 
                            ground_truth: List[Dict[str, Any]], 
                            k: int = 32) -> Dict[str, float]:
        """
        Evaluate trajectories against ground truth
        
        Args:
            trajectories: List of trajectory dictionaries with questions and attempts
            ground_truth: List of ground truth dictionaries
            k: Number of attempts to consider for Mean@K
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create mapping from question to ground truth answer
        gt_mapping = {}
        for gt_item in ground_truth:
            question = gt_item.get('question', '')
            answer = gt_item.get('answer', '')
            gt_mapping[question] = answer
        
        predictions_list = []
        ground_truths_list = []
        single_predictions = []
        
        matched_count = 0
        
        for trajectory in trajectories:
            question = trajectory.get('question', '')
            attempts = trajectory.get('attempts', [])
            
            if question in gt_mapping:
                matched_count += 1
                gt_answer = gt_mapping[question]
                
                # Extract responses from attempts
                responses = []
                for attempt in attempts:
                    response = attempt.get('response', '') or attempt.get('reasoning_steps', '')
                    responses.append(response)
                
                predictions_list.append(responses)
                ground_truths_list.append(gt_answer)
                
                # For accuracy calculation, use first response
                if responses:
                    single_predictions.append(responses[0])
                else:
                    single_predictions.append('')
        
        logger.info(f"Matched {matched_count} questions out of {len(trajectories)} trajectories")
        
        if not predictions_list:
            logger.warning("No matching questions found between trajectories and ground truth")
            return {
                'mean_at_k': 0.0,
                'accuracy': 0.0,
                'total_questions': len(trajectories),
                'matched_questions': 0
            }
        
        # Calculate metrics
        mean_k_score = mean_at_k(predictions_list, ground_truths_list, k)
        accuracy_score = calculate_accuracy(single_predictions, ground_truths_list)
        
        results = {
            'mean_at_k': mean_k_score,
            f'mean_at_{k}': mean_k_score,  # Explicit k value
            'accuracy': accuracy_score,
            'mean_at_1': accuracy_score,  # Accuracy is Mean@1
            'total_questions': len(trajectories),
            'matched_questions': matched_count,
            'k_value': k
        }
        
        self.results.append(results)
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Mean@{k}: {mean_k_score:.4f}")
        logger.info(f"  Accuracy (Mean@1): {accuracy_score:.4f}")
        logger.info(f"  Matched Questions: {matched_count}/{len(trajectories)}")
        
        return results
    
    def evaluate_files(self, predictions_file: Union[str, Path], 
                      ground_truth_file: Union[str, Path], 
                      k: int = 32) -> Dict[str, float]:
        """
        Evaluate predictions file against ground truth file
        
        Args:
            predictions_file: Path to predictions JSON file
            ground_truth_file: Path to ground truth JSON file  
            k: Number of predictions to consider for Mean@K
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.load_predictions(predictions_file)
        ground_truth = self.load_ground_truth(ground_truth_file)
        
        return self.evaluate_trajectories(predictions, ground_truth, k)
    
    def save_results(self, output_file: Union[str, Path]):
        """
        Save evaluation results to file
        
        Args:
            output_file: Path to output JSON file
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {output_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all evaluation results
        
        Returns:
            Summary dictionary
        """
        if not self.results:
            return {}
        
        # Calculate averages across all evaluations
        total_mean_k = sum(r['mean_at_k'] for r in self.results)
        total_accuracy = sum(r['accuracy'] for r in self.results)
        count = len(self.results)
        
        return {
            'num_evaluations': count,
            'avg_mean_at_k': total_mean_k / count,
            'avg_accuracy': total_accuracy / count,
            'all_results': self.results
        }