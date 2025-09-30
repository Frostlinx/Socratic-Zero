"""
DPO Data Converter

Converts judge results to DPO training format.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DPODataConverter:
    """
    DPO data converter for creating training datasets
    """
    
    def __init__(self):
        self.workspace_dir = Path(os.getenv("WORKSPACE_DIR", "/workspace/prosetting"))
        
    def create_dataset_for_round(self, round_num: int, judge_results: List[Dict[str, Any]]) -> str:
        """
        Create DPO dataset for training round
        
        Args:
            round_num: Training round number
            judge_results: Judge results from reward calculation
            
        Returns:
            Path to created dataset directory
        """
        dataset_dir = self.workspace_dir / "datasets" / f"round_{round_num}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to DPO format
        dpo_data = self._convert_to_dpo_format(judge_results)
        
        # Split into train/validation
        train_data, val_data = self._split_data(dpo_data)
        
        # Save as parquet files (placeholder - would use pandas/pyarrow in real implementation)
        train_file = dataset_dir / "train.parquet"
        val_file = dataset_dir / "validation.parquet"
        
        # Placeholder save logic
        self._save_as_json(train_data, dataset_dir / "train.json")
        self._save_as_json(val_data, dataset_dir / "validation.json")
        
        # Create dataset info
        dataset_info = {
            "round": round_num,
            "total_samples": len(dpo_data),
            "train_samples": len(train_data),
            "validation_samples": len(val_data),
            "format": "DPO",
            "files": {
                "train": str(train_file),
                "validation": str(val_file)
            }
        }
        
        info_file = dataset_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created DPO dataset for round {round_num}: {dataset_dir}")
        return str(dataset_dir)
    
    def _convert_to_dpo_format(self, judge_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert judge results to DPO triplet format"""
        dpo_data = []
        
        for result in judge_results:
            # Create DPO triplet (prompt, chosen, rejected)
            dpo_item = {
                "prompt": result.get("question", ""),
                "chosen": result.get("correct_answers", [""])[0] if result.get("correct_answers") else "",
                "rejected": result.get("incorrect_answers", [""])[0] if result.get("incorrect_answers") else "",
                "reward": result.get("reward", 0.0)
            }
            dpo_data.append(dpo_item)
        
        return dpo_data
    
    def _split_data(self, data: List[Dict[str, Any]], train_ratio: float = 0.8) -> tuple:
        """Split data into train and validation sets"""
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]
    
    def _save_as_json(self, data: List[Dict[str, Any]], file_path: Path):
        """Save data as JSON file (placeholder for parquet)"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_dataset_stats(self, dataset_dir: str) -> Dict[str, Any]:
        """Get dataset statistics"""
        dataset_path = Path(dataset_dir)
        info_file = dataset_path / "dataset_info.json"
        
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {"error": "Dataset info not found"}