"""
Data Saver

Saves training data in various formats.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DataSaver:
    """
    Data saver for training data persistence
    """
    
    def __init__(self):
        self.workspace_dir = Path(os.getenv("WORKSPACE_DIR", "/workspace/prosetting"))
        
    def save_training_data(self, data: List[Dict[str, Any]], 
                          round_num: int, data_type: str) -> str:
        """
        Save training data to file
        
        Args:
            data: Data to save
            round_num: Training round number
            data_type: Type of data (trajectories, results, etc.)
            
        Returns:
            Path to saved file
        """
        data_dir = self.workspace_dir / "training_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"round_{round_num}_{data_type}.json"
        file_path = data_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {data_type} data to: {file_path}")
        return str(file_path)
    
    def load_training_data(self, round_num: int, data_type: str) -> List[Dict[str, Any]]:
        """
        Load training data from file
        
        Args:
            round_num: Training round number
            data_type: Type of data to load
            
        Returns:
            Loaded data
        """
        data_dir = self.workspace_dir / "training_data"
        filename = f"round_{round_num}_{data_type}.json"
        file_path = data_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Training data file not found: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} items from: {file_path}")
        return data