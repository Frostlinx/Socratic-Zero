"""
Solver Data Processor

Processes and saves solver training data and results.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SolverDataProcessor:
    """
    Solver data processor for saving and managing training data
    """
    
    def __init__(self):
        self.workspace_dir = Path(os.getenv("WORKSPACE_DIR", "/workspace/prosetting"))
        
    def save_judge_results(self, judge_results: List[Dict[str, Any]], round_num: int) -> str:
        """
        Save judge results to file
        
        Args:
            judge_results: List of judge result dictionaries
            round_num: Training round number
            
        Returns:
            Path to saved file
        """
        data_dir = self.workspace_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        judge_file = data_dir / f"round_{round_num}_judge_results.json"
        
        with open(judge_file, 'w', encoding='utf-8') as f:
            json.dump(judge_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved judge results to: {judge_file}")
        return str(judge_file)
    
    def load_judge_results(self, round_num: int) -> List[Dict[str, Any]]:
        """
        Load judge results from file
        
        Args:
            round_num: Training round number
            
        Returns:
            List of judge result dictionaries
        """
        data_dir = self.workspace_dir / "data"
        judge_file = data_dir / f"round_{round_num}_judge_results.json"
        
        if not judge_file.exists():
            logger.warning(f"Judge results file not found: {judge_file}")
            return []
        
        with open(judge_file, 'r', encoding='utf-8') as f:
            judge_results = json.load(f)
        
        logger.info(f"Loaded {len(judge_results)} judge results from: {judge_file}")
        return judge_results