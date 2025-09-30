"""
Question Manager

Manages question loading and processing for training rounds.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class QuestionManager:
    """
    Question manager for training data
    """
    
    @staticmethod
    def load_questions_from_file(file_path: str) -> List[Dict[str, Any]]:
        """
        Load questions from JSON file
        
        Args:
            file_path: Path to questions file
            
        Returns:
            List of question dictionaries
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Questions file not found: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            logger.info(f"Loaded {len(questions)} questions from {file_path}")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to load questions from {file_path}: {e}")
            return []
    
    @staticmethod
    def validate_questions(questions: List[Dict[str, Any]]) -> bool:
        """
        Validate question format
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            True if questions are valid
        """
        if not questions:
            logger.warning("No questions provided for validation")
            return False
        
        required_fields = ["question"]
        
        for i, question in enumerate(questions):
            if not isinstance(question, dict):
                logger.error(f"Question {i} is not a dictionary")
                return False
            
            for field in required_fields:
                if field not in question:
                    logger.error(f"Question {i} missing required field: {field}")
                    return False
        
        logger.info(f"Validated {len(questions)} questions successfully")
        return True
    
    @staticmethod
    def filter_questions(questions: List[Dict[str, Any]], 
                        criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Filter questions based on criteria
        
        Args:
            questions: List of question dictionaries
            criteria: Filtering criteria
            
        Returns:
            Filtered list of questions
        """
        if not criteria:
            return questions
        
        filtered = []
        for question in questions:
            # Placeholder filtering logic
            # In real implementation, would apply actual filtering criteria
            filtered.append(question)
        
        logger.info(f"Filtered {len(questions)} questions to {len(filtered)}")
        return filtered