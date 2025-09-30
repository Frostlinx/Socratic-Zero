"""
Evaluation Metrics

Common evaluation metrics for math problem solving
"""

import re
from typing import List, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def extract_final_answer(text: str) -> str:
    """
    Extract final answer from model response
    
    Args:
        text: Model response text
        
    Returns:
        Extracted final answer string
    """
    # Look for patterns like "Final Answer: X" or "Answer: X"
    patterns = [
        r'Final Answer[：:]\s*([^\n]+)',
        r'Answer[：:]\s*([^\n]+)',
        r'The answer is[：:]\s*([^\n]+)',
        r'Result[：:]\s*([^\n]+)',
        r'Solution[：:]\s*([^\n]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no pattern found, try to extract number from the end
    lines = text.strip().split('\n')
    for line in reversed(lines):
        # Look for numbers in the last few lines
        numbers = re.findall(r'-?\d+\.?\d*', line)
        if numbers:
            return numbers[-1]
    
    return text.strip()


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer string
    """
    if not answer:
        return ""
    
    # Remove common prefixes and suffixes
    answer = re.sub(r'^(The answer is|Answer is|equals|=)\s*', '', answer)
    answer = re.sub(r'\s*(yuan|units|meters|minutes|seconds|days|years)
    
    # Extract numbers and basic operations
    # Handle fractions like 1/2, decimals like 3.14, integers like 42
    number_match = re.search(r'-?\d+(?:\.\d+)?(?:/\d+)?', answer)
    if number_match:
        return number_match.group(0)
    
    return answer.strip().lower()


def is_correct(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted answer matches ground truth
    
    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        True if answers match, False otherwise
    """
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    if pred_norm == gt_norm:
        return True
    
    # Try numerical comparison for different formats
    try:
        # Handle fractions
        if '/' in pred_norm and '/' in gt_norm:
            pred_parts = pred_norm.split('/')
            gt_parts = gt_norm.split('/')
            if len(pred_parts) == 2 and len(gt_parts) == 2:
                pred_val = float(pred_parts[0]) / float(pred_parts[1])
                gt_val = float(gt_parts[0]) / float(gt_parts[1])
                return abs(pred_val - gt_val) < 1e-6
        
        # Handle decimal comparison
        pred_float = float(pred_norm)
        gt_float = float(gt_norm)
        return abs(pred_float - gt_float) < 1e-6
        
    except (ValueError, ZeroDivisionError):
        pass
    
    return False


def mean_at_k(predictions: List[List[str]], ground_truths: List[str], k: int = 32) -> float:
    """
    Calculate Mean@K metric
    
    Args:
        predictions: List of prediction lists for each question (each inner list has k predictions)
        ground_truths: List of ground truth answers
        k: Number of predictions to consider (default 32)
        
    Returns:
        Mean@K score (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Predictions length ({len(predictions)}) != ground truths length ({len(ground_truths)})")
    
    correct_count = 0
    total_count = len(predictions)
    
    for pred_list, gt in zip(predictions, ground_truths):
        # Take first k predictions
        k_predictions = pred_list[:k]
        
        # Check if any of the k predictions is correct
        is_any_correct = False
        for pred in k_predictions:
            extracted_pred = extract_final_answer(pred)
            if is_correct(extracted_pred, gt):
                is_any_correct = True
                break
        
        if is_any_correct:
            correct_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate simple accuracy (Mean@1)
    
    Args:
        predictions: List of single predictions
        ground_truths: List of ground truth answers
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Predictions length ({len(predictions)}) != ground truths length ({len(ground_truths)})")
    
    correct_count = 0
    for pred, gt in zip(predictions, ground_truths):
        extracted_pred = extract_final_answer(pred)
        if is_correct(extracted_pred, gt):
            correct_count += 1
    
    return correct_count / len(predictions) if len(predictions) > 0 else 0.0, '', answer)
    
    # Extract numbers and basic operations
    # Handle fractions like 1/2, decimals like 3.14, integers like 42
    number_match = re.search(r'-?\d+(?:\.\d+)?(?:/\d+)?', answer)
    if number_match:
        return number_match.group(0)
    
    return answer.strip().lower()


def is_correct(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted answer matches ground truth
    
    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        True if answers match, False otherwise
    """
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    if pred_norm == gt_norm:
        return True
    
    # Try numerical comparison for different formats
    try:
        # Handle fractions
        if '/' in pred_norm and '/' in gt_norm:
            pred_parts = pred_norm.split('/')
            gt_parts = gt_norm.split('/')
            if len(pred_parts) == 2 and len(gt_parts) == 2:
                pred_val = float(pred_parts[0]) / float(pred_parts[1])
                gt_val = float(gt_parts[0]) / float(gt_parts[1])
                return abs(pred_val - gt_val) < 1e-6
        
        # Handle decimal comparison
        pred_float = float(pred_norm)
        gt_float = float(gt_norm)
        return abs(pred_float - gt_float) < 1e-6
        
    except (ValueError, ZeroDivisionError):
        pass
    
    return False


def mean_at_k(predictions: List[List[str]], ground_truths: List[str], k: int = 32) -> float:
    """
    Calculate Mean@K metric
    
    Args:
        predictions: List of prediction lists for each question (each inner list has k predictions)
        ground_truths: List of ground truth answers
        k: Number of predictions to consider (default 32)
        
    Returns:
        Mean@K score (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Predictions length ({len(predictions)}) != ground truths length ({len(ground_truths)})")
    
    correct_count = 0
    total_count = len(predictions)
    
    for pred_list, gt in zip(predictions, ground_truths):
        # Take first k predictions
        k_predictions = pred_list[:k]
        
        # Check if any of the k predictions is correct
        is_any_correct = False
        for pred in k_predictions:
            extracted_pred = extract_final_answer(pred)
            if is_correct(extracted_pred, gt):
                is_any_correct = True
                break
        
        if is_any_correct:
            correct_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate simple accuracy (Mean@1)
    
    Args:
        predictions: List of single predictions
        ground_truths: List of ground truth answers
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Predictions length ({len(predictions)}) != ground truths length ({len(ground_truths)})")
    
    correct_count = 0
    for pred, gt in zip(predictions, ground_truths):
        extracted_pred = extract_final_answer(pred)
        if is_correct(extracted_pred, gt):
            correct_count += 1
    
    return correct_count / len(predictions) if len(predictions) > 0 else 0.0