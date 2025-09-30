"""
Round Controller

Controls training round progression and management.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RoundController:
    """
    Training round controller
    """
    
    def __init__(self, max_rounds: int = 10, save_rounds: List[int] = None):
        """
        Initialize round controller
        
        Args:
            max_rounds: Maximum number of training rounds
            save_rounds: List of rounds to save checkpoints
        """
        self.max_rounds = max_rounds
        self.save_rounds = save_rounds or [3, 4, 5, 6, 7, 8, 9, 10]
        self.current_round = 1
        
    def should_save_checkpoint(self, round_num: int) -> bool:
        """
        Check if checkpoint should be saved for this round
        
        Args:
            round_num: Current round number
            
        Returns:
            True if checkpoint should be saved
        """
        return round_num in self.save_rounds
    
    def is_final_round(self, round_num: int) -> bool:
        """
        Check if this is the final round
        
        Args:
            round_num: Current round number
            
        Returns:
            True if this is the final round
        """
        return round_num >= self.max_rounds
    
    def get_round_info(self, round_num: int) -> Dict[str, Any]:
        """
        Get information about a specific round
        
        Args:
            round_num: Round number
            
        Returns:
            Dictionary with round information
        """
        return {
            "round_num": round_num,
            "max_rounds": self.max_rounds,
            "should_save": self.should_save_checkpoint(round_num),
            "is_final": self.is_final_round(round_num),
            "progress": round_num / self.max_rounds
        }