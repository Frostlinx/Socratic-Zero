"""
Data Normalizer

Normalizes collected trajectory data for consistent processing.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Data normalizer for trajectory data
    """
    
    @staticmethod
    def normalize_trajectories(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize trajectory data format
        
        Args:
            trajectories: Raw trajectory data
            
        Returns:
            Normalized trajectory data
        """
        normalized = []
        
        for trajectory in trajectories:
            normalized_trajectory = {
                "question": trajectory.get("question", ""),
                "attempt": trajectory.get("attempt", 1),
                "response": trajectory.get("response", ""),
                "reasoning_steps": trajectory.get("reasoning_steps", ""),
                "metadata": {
                    "original_keys": list(trajectory.keys()),
                    "normalized": True
                }
            }
            normalized.append(normalized_trajectory)
        
        logger.info(f"Normalized {len(normalized)} trajectories")
        return normalized