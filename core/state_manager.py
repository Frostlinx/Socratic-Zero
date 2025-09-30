"""
State Manager

Manages training state, progress, and checkpoint data.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import datetime

logger = logging.getLogger(__name__)


class StateManager:
    """
    Training state manager for checkpoint and progress tracking
    """
    
    def __init__(self, workspace_dir: str = None):
        """
        Initialize state manager
        
        Args:
            workspace_dir: Workspace directory path
        """
        self.workspace_dir = Path(workspace_dir or os.getenv("WORKSPACE_DIR", "/workspace/prosetting"))
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.workspace_dir / "training_state.json"
        self.config_file = self.workspace_dir / "training_config.json"
        
    def load_training_state(self) -> Optional[Dict[str, Any]]:
        """
        Load training state from file
        
        Returns:
            Training state dictionary or None if not found
        """
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            logger.info("Training state loaded successfully")
            return state
        except Exception as e:
            logger.error(f"Failed to load training state: {e}")
            return None
    
    def save_training_state(self, state: Dict[str, Any]):
        """
        Save training state to file
        
        Args:
            state: Training state dictionary
        """
        try:
            state["last_updated"] = datetime.datetime.now().isoformat()
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            logger.info("Training state saved successfully")
        except Exception as e:
            logger.error(f"Failed to save training state: {e}")
    
    def load_training_config(self) -> Optional[Dict[str, Any]]:
        """
        Load training configuration
        
        Returns:
            Training configuration dictionary or None if not found
        """
        if not self.config_file.exists():
            return None
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("Training configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load training configuration: {e}")
            return None
    
    def save_training_config(self, config: Dict[str, Any]):
        """
        Save training configuration
        
        Args:
            config: Training configuration dictionary
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info("Training configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save training configuration: {e}")
    
    def get_current_round(self) -> int:
        """
        Get current training round
        
        Returns:
            Current round number (1-based)
        """
        state = self.load_training_state()
        if state:
            return state.get("current_round", 1)
        return 1
    
    def get_completed_rounds(self) -> List[int]:
        """
        Get list of completed rounds
        
        Returns:
            List of completed round numbers
        """
        state = self.load_training_state()
        if state:
            return state.get("completed_rounds", [])
        return []
    
    def mark_round_completed(self, round_num: int, success: bool = True):
        """
        Mark a round as completed
        
        Args:
            round_num: Round number to mark as completed
            success: Whether the round completed successfully
        """
        state = self.load_training_state() or {}
        
        completed_rounds = state.get("completed_rounds", [])
        if round_num not in completed_rounds and success:
            completed_rounds.append(round_num)
        
        state.update({
            "completed_rounds": completed_rounds,
            "current_round": round_num + 1,
            "last_completed_round": round_num if success else state.get("last_completed_round", 0)
        })
        
        self.save_training_state(state)
    
    def get_round_status(self, round_num: int) -> Dict[str, Any]:
        """
        Get status of a specific round
        
        Args:
            round_num: Round number
            
        Returns:
            Dictionary with round status information
        """
        completed_rounds = self.get_completed_rounds()
        current_round = self.get_current_round()
        
        if round_num in completed_rounds:
            status = "completed"
            fully_completed = True
        elif round_num == current_round:
            status = "in_progress"
            fully_completed = False
        elif round_num < current_round:
            status = "completed"
            fully_completed = True
        else:
            status = "pending"
            fully_completed = False
        
        return {
            "round_num": round_num,
            "status": status,
            "fully_completed": fully_completed,
            "completed_stages": [],
            "next_stage": "data_collection" if status == "pending" else None
        }
    
    def is_stage_completed(self, round_num: int, stage: str) -> bool:
        """
        Check if a specific stage is completed for a round
        
        Args:
            round_num: Round number
            stage: Stage name
            
        Returns:
            True if stage is completed
        """
        # Placeholder implementation
        # In real implementation, would check detailed stage progress
        return False
    
    def save_round_progress(self, round_num: int, stage: str, data: Dict[str, Any]):
        """
        Save progress for a specific round and stage
        
        Args:
            round_num: Round number
            stage: Stage name
            data: Progress data
        """
        progress_file = self.workspace_dir / f"round_{round_num}_progress.json"
        
        # Load existing progress
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
        else:
            progress = {"round": round_num, "stages": {}}
        
        # Update stage progress
        progress["stages"][stage] = data
        progress["last_updated"] = datetime.datetime.now().isoformat()
        
        # Save progress
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved progress for round {round_num}, stage {stage}")
    
    def load_stage_data(self, round_num: int, stage: str) -> Optional[Dict[str, Any]]:
        """
        Load data for a specific round and stage
        
        Args:
            round_num: Round number
            stage: Stage name
            
        Returns:
            Stage data or None if not found
        """
        progress_file = self.workspace_dir / f"round_{round_num}_progress.json"
        
        if not progress_file.exists():
            return None
        
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            return progress.get("stages", {}).get(stage)
        except Exception as e:
            logger.error(f"Failed to load stage data: {e}")
            return None
    
    def save_round_data(self, round_num: int, data_type: str, data: Dict[str, Any]):
        """
        Save round-specific data
        
        Args:
            round_num: Round number
            data_type: Type of data
            data: Data to save
        """
        data_dir = self.workspace_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        data_file = data_dir / f"round_{round_num}_{data_type}.json"
        
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved round {round_num} {data_type} data")
    
    def load_round_data(self, round_num: int, data_type: str) -> Optional[Dict[str, Any]]:
        """
        Load round-specific data
        
        Args:
            round_num: Round number
            data_type: Type of data
            
        Returns:
            Round data or None if not found
        """
        data_dir = self.workspace_dir / "data"
        data_file = data_dir / f"round_{round_num}_{data_type}.json"
        
        if not data_file.exists():
            return None
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load round data: {e}")
            return None
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """
        Get workspace summary information
        
        Returns:
            Dictionary with workspace summary
        """
        data_dir = self.workspace_dir / "data"
        
        summary = {
            "workspace_dir": str(self.workspace_dir),
            "state_exists": self.state_file.exists(),
            "config_exists": self.config_file.exists(),
            "total_data_files": len(list(data_dir.glob("*.json"))) if data_dir.exists() else 0,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        return summary