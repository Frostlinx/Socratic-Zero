#!/usr/bin/env python3
"""
Semi-Automated TRL Training Script
Uses TRL DPO instead of VERL PPO with checkpoint recovery
"""

import os
import sys
import time
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Setup project paths
project_root = Path(__file__).parent.parent.absolute()
prosetting_root = project_root / "ProSetting"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(prosetting_root) not in sys.path:
    sys.path.insert(0, str(prosetting_root))

# Load environment variables
env_file = prosetting_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ Environment variables loaded from {env_file}")
else:
    print(f"⚠️ Environment file not found: {env_file}")

# Import modular components
from collectors.trajectory_collector import TrajectoryCollector
from collectors.data_normalizer import DataNormalizer
from processors.reward_calculator import RewardCalculator
from processors.question_enhancer import QuestionEnhancer
from processors.solver_data_processor import SolverDataProcessor
from datasets.dpo_data_converter import DPODataConverter
from datasets.data_saver import DataSaver
from trainers.trl_trainer import TRLTrainer
from trainers.gpu_manager import GPUManager
from managers.round_controller import RoundController
from managers.question_manager import QuestionManager
from core.state_manager import StateManager

# Configure logging
log_dir = os.getenv('LOG_DIR', '/workspace/prosetting/logs')
log_dir_path = Path(log_dir)
log_dir_path.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir_path / 'trl_trainer.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file_path), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class SemiAutoTRLTrainer:
    """Semi-automated TRL trainer using DPO instead of PPO"""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.round_controller = None
        self.modules = {}
        self.config = {}
        
    def initialize(self):
        """Initialize training environment"""
        print("🚀 ========== Semi-Automated TRL Training System Initialization ==========")
        
        # Setup GPU environment
        GPUManager.setup_gpu_environment()
        
        # Load or create configuration
        self.config = self.state_manager.load_training_config()
        if not self.config:
            self.config = self._create_default_config()
            self.state_manager.save_training_config(self.config)
        
        # Initialize round controller
        self.round_controller = RoundController(
            max_rounds=self.config["max_rounds"],
            save_rounds=self.config["save_rounds"]
        )
        
        # Initialize modules
        self._initialize_modules()
        
        print(f"✅ TRL training system initialization completed")
        print(f"📊 Configuration info:")
        print(f"   - Total rounds: {self.config['max_rounds']}")
        print(f"   - Save rounds: {self.config['save_rounds']}")
        print(f"   - Training framework: TRL DPO")
        print(f"   - Workspace: {self.state_manager.workspace_dir}")
        
        return True
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration from .env file"""
        logger.info("🔍 Loading TRL configuration from .env file...")
        
        # Basic configuration from environment variables
        solver_model_path = os.getenv("SOLVER_MODEL_PATH")
        questions_file = os.getenv("QUESTIONS_FILE")
        workspace_dir = os.getenv("WORKSPACE_DIR")
        
        # Validate required environment variables
        if not solver_model_path:
            raise ValueError("SOLVER_MODEL_PATH environment variable is required")
        if not questions_file:
            raise ValueError("QUESTIONS_FILE environment variable is required")
        if not workspace_dir:
            raise ValueError("WORKSPACE_DIR environment variable is required")
        
        # Training parameters
        max_rounds = int(os.getenv("TOTAL_ROUNDS", "10"))
        save_rounds_str = os.getenv("SAVE_ROUNDS", "3,4,5,6,7,8,9,10")
        save_rounds = [int(x.strip()) for x in save_rounds_str.split(",")]
        attempts_per_question = int(os.getenv("ATTEMPTS_PER_QUESTION", "8"))
        
        # GPU configuration
        physical_solver_gpu = os.getenv("PHYSICAL_SOLVER_GPU", "4")
        physical_dpo_gpu = os.getenv("PHYSICAL_DPO_GPU", "0,1,2,3,4,5,6,7")
        
        # Teacher model configuration
        teacher_base_url = os.getenv("TEACHER_BASE_URL", "http://localhost:8000")
        teacher_concurrent_workers = int(os.getenv("TEACHER_CONCURRENT_WORKERS", "32"))
        
        # Validate questions file exists
        if not os.path.exists(questions_file):
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
        config = {
            "max_rounds": max_rounds,
            "save_rounds": save_rounds,
            "attempts_per_question": attempts_per_question,
            "solver_model_path": solver_model_path,
            "questions_file": questions_file,
            "workspace_dir": workspace_dir,
            "physical_solver_gpu": physical_solver_gpu,
            "physical_dpo_gpu": physical_dpo_gpu,
            "teacher_base_url": teacher_base_url,
            "teacher_concurrent_workers": teacher_concurrent_workers,
            "training_framework": "TRL_DPO"
        }
        
        logger.info(f"🔍 TRL Configuration loaded:")
        logger.info(f"   - Training framework: TRL DPO")
        logger.info(f"   - Solver model: {solver_model_path}")
        logger.info(f"   - Max rounds: {max_rounds}")
        logger.info(f"   - Save rounds: {save_rounds}")
        logger.info(f"   - Questions file: {questions_file}")
        
        return config
    
    def _initialize_modules(self):
        """Initialize all modules"""
        # Placeholder for teacher clients - implement based on your teacher model setup
        teacher1_client = None  # get_teacher1_client()
        teacher2_client = None  # get_teacher2_client()
        
        self.modules = {
            "trajectory_collector": TrajectoryCollector(physical_gpus=self.config["physical_solver_gpu"]),
            "reward_calculator": RewardCalculator(),
            "question_enhancer": QuestionEnhancer(),
            "dpo_converter": DPODataConverter(),
            "solver_data_processor": SolverDataProcessor(),
            "trl_trainer": TRLTrainer(save_rounds=self.config["save_rounds"])
        }
        
        self.modules["reward_calculator"].set_teacher_client(teacher1_client)
        self.modules["question_enhancer"].set_teacher_client(teacher2_client)
    
    def execute_round(self, round_num: int) -> Dict[str, Any]:
        """Execute single TRL training round"""
        print(f"\n🚀 ========== Executing Round {round_num} TRL Training ==========")
        
        # Check round status
        round_status = self.state_manager.get_round_status(round_num)
        
        if round_status["fully_completed"]:
            print(f"⚠️ Round {round_num} already completed, skipping execution")
            return {"success": True, "skipped": True, "message": "Round already completed"}
        
        try:
            # Get all questions
            all_questions = self._get_round_questions(round_num)
            if not all_questions:
                return {"success": False, "message": "Failed to load questions"}
            
            print(f"📊 Total {len(all_questions)} questions to process")
            
            # Stage 1: Data collection
            if not self.state_manager.is_stage_completed(round_num, "data_collection"):
                print("🚀 Stage 1: Data collection")
                collection_result = self._execute_data_collection(round_num, all_questions)
                if not collection_result["success"]:
                    return collection_result
                
                self.state_manager.save_round_progress(round_num, "data_collection", {
                    "trajectories": collection_result["data"]["all_trajectories"],
                    "questions": all_questions,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            else:
                print("✅ Stage 1: Data collection completed, loading from checkpoint")
                collection_data = self.state_manager.load_stage_data(round_num, "data_collection")
                collection_result = {
                    "success": True,
                    "data": {
                        "all_trajectories": collection_data["trajectories"],
                        "questions": collection_data["questions"]
                    }
                }
            
            # Stage 2: Data judging
            if not self.state_manager.is_stage_completed(round_num, "data_judging"):
                print("🚀 Stage 2: Data judging")
                judging_result = self._execute_data_judging(round_num, collection_result["data"]["all_trajectories"])
                if not judging_result["success"]:
                    return judging_result
                
                self.state_manager.save_round_progress(round_num, "data_judging", {
                    "judge_results": judging_result["data"]["judge_results"],
                    "timestamp": datetime.datetime.now().isoformat()
                })
            else:
                print("✅ Stage 2: Data judging completed, loading from checkpoint")
                judging_data = self.state_manager.load_stage_data(round_num, "data_judging")
                judging_result = {
                    "success": True,
                    "data": {"judge_results": judging_data["judge_results"]}
                }
            
            # Stage 3: DPO data conversion
            if not self.state_manager.is_stage_completed(round_num, "dpo_conversion"):
                print("🚀 Stage 3: DPO data conversion")
                conversion_result = self._execute_dpo_conversion(round_num, judging_result["data"]["judge_results"])
                if not conversion_result["success"]:
                    return conversion_result
                
                self.state_manager.save_round_progress(round_num, "dpo_conversion", {
                    "dataset_dir": conversion_result["data"]["dataset_dir"],
                    "timestamp": datetime.datetime.now().isoformat()
                })
            else:
                print("✅ Stage 3: DPO data conversion completed, loading from checkpoint")
                conversion_data = self.state_manager.load_stage_data(round_num, "dpo_conversion")
                conversion_result = {
                    "success": True,
                    "data": {"dataset_dir": conversion_data["dataset_dir"]}
                }
            
            # Stage 4: TRL training - skip first 2 rounds, start from round 3
            if round_num <= 2:
                print(f"⏭️ Stage 4: Skip TRL training (first 2 rounds, current: {round_num})")
                training_result = {"success": True, "skipped": True}
                self.state_manager.save_round_progress(round_num, "trl_training", {
                    "training_success": True,
                    "skipped": True,
                    "reason": f"Round {round_num} <= 2, no training in first 2 rounds",
                    "timestamp": datetime.datetime.now().isoformat()
                })
            else:
                if not self.state_manager.is_stage_completed(round_num, "trl_training"):
                    print(f"🚀 Stage 4: TRL training (round {round_num} >= 3, start training)")
                    training_result = self._execute_trl_training(round_num, conversion_result["data"]["dataset_dir"])
                    
                    self.state_manager.save_round_progress(round_num, "trl_training", {
                        "training_success": training_result.get("success", False),
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                else:
                    print("✅ Stage 4: TRL training completed")
                    training_result = {"success": True}
            
            # Stage 5: Prepare next round
            if not self.state_manager.is_stage_completed(round_num, "next_round_prep"):
                print("🚀 Stage 5: Prepare next round")
                self._prepare_next_round(round_num, all_questions, judging_result["data"]["judge_results"])
                
                self.state_manager.save_round_progress(round_num, "next_round_prep", {
                    "preparation_complete": True,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            else:
                print("✅ Stage 5: Next round preparation completed")
            
            # Check if training was successful
            training_success = training_result.get("success", False)
            
            if not training_success and not training_result.get("skipped", False):
                print(f"⚠️ Round {round_num} TRL training failed, not proceeding to next round")
                return {
                    "success": False,
                    "round_num": round_num,
                    "message": f"Round {round_num} TRL training failed",
                    "can_retry": True
                }
            
            # Mark round as completed
            self.state_manager.mark_round_completed(round_num, True)
            
            print(f"✅ Round {round_num} TRL training completed successfully")
            return {
                "success": True,
                "round_num": round_num,
                "total_questions": len(all_questions),
                "message": f"Round {round_num} TRL training completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Round {round_num} execution failed: {e}", exc_info=True)
            
            self.state_manager.save_round_progress(round_num, "execution_error", {
                "error_message": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return {
                "success": False, 
                "message": f"Execution error: {str(e)}",
                "can_retry": True,
                "error_type": "execution_error"
            }
    
    def _execute_data_collection(self, round_num: int, all_questions: list) -> Dict[str, Any]:
        """Execute data collection"""
        try:
            collector = self.modules["trajectory_collector"]
            solver_path = self._get_solver_path(round_num)
            
            # Load model
            success = collector.load_solver_model(solver_path, force_load=(round_num == 1))
            if not success:
                return {"success": False, "message": "Failed to load solver model"}
            
            # Collect trajectories
            trajectories = collector.collect_trajectories(
                all_questions, attempts_per_question=self.config["attempts_per_question"]
            )
            
            if not trajectories:
                return {"success": False, "message": "No trajectories collected"}
            
            normalized_trajectories = DataNormalizer.normalize_trajectories(trajectories)
            
            print(f"✅ Data collection completed, collected {len(normalized_trajectories)} trajectories")
            
            return {
                "success": True,
                "data": {"all_trajectories": normalized_trajectories}
            }
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}", exc_info=True)
            return {"success": False, "message": f"Data collection error: {str(e)}"}
    
    def _execute_data_judging(self, round_num: int, trajectories: list) -> Dict[str, Any]:
        """Execute data judging"""
        try:
            print("📊 Executing judging calculation...")
            
            rewards, judge_results = self.modules["reward_calculator"].compute_solver_reward_local(trajectories)
            
            # Save judge results to file
            processor = self.modules["solver_data_processor"]
            judge_file = processor.save_judge_results(judge_results, round_num)
            
            print(f"✅ Judging completed, average reward: {sum(rewards)/len(rewards):.3f}")
            print(f"💾 Judge results saved: {judge_file}")
            
            return {
                "success": True,
                "data": {"judge_results": judge_results}
            }
            
        except Exception as e:
            logger.error(f"Data judging failed: {e}", exc_info=True)
            return {"success": False, "message": f"Data judging error: {str(e)}"}
    
    def _execute_dpo_conversion(self, round_num: int, judge_results: list) -> Dict[str, Any]:
        """Execute DPO data conversion"""
        try:
            print("🔄 Converting judge results to DPO format...")
            
            converter = self.modules["dpo_converter"]
            dataset_dir = converter.create_dataset_for_round(round_num, judge_results)
            
            # Get dataset statistics
            stats = converter.get_dataset_stats(dataset_dir)
            
            print(f"✅ DPO data conversion completed")
            print(f"📊 Training samples: {stats['files']['train']['samples']}")
            print(f"📊 Validation samples: {stats['files']['validation']['samples']}")
            print(f"💾 Dataset directory: {dataset_dir}")
            
            return {
                "success": True,
                "data": {"dataset_dir": dataset_dir, "stats": stats}
            }
            
        except Exception as e:
            logger.error(f"DPO conversion failed: {e}", exc_info=True)
            return {"success": False, "message": f"DPO conversion error: {str(e)}"}
    
    def _execute_trl_training(self, round_num: int, dataset_dir: str) -> Dict[str, Any]:
        """Execute TRL training"""
        try:
            print(f"🚀 Starting TRL DPO training...")
            
            # Release collector model and clear GPU memory
            if hasattr(self.modules["trajectory_collector"], '_model_loaded'):
                self.modules["trajectory_collector"].release_model()
            
            GPUManager.clear_gpu_memory()
            GPUManager.set_gpu_environment(self.config["physical_dpo_gpu"])
            
            # Execute TRL training
            trainer = self.modules["trl_trainer"]
            success = trainer.run_trl_training(dataset_dir, round_num)
            
            GPUManager.cleanup_and_release_models()
            
            if success:
                print(f"🎉 TRL training completed successfully")
            else:
                print(f"❌ TRL training failed")
            
            return {"success": success}
            
        except Exception as e:
            logger.error(f"TRL training failed: {e}", exc_info=True)
            return {"success": False, "message": f"TRL training error: {str(e)}"}
    
    def _prepare_next_round(self, round_num: int, current_questions: list, judge_results: list):
        """Prepare next round data"""
        try:
            # Extract failed questions from judge results
            failed_questions = []
            for result in judge_results:
                if result.get("incorrect_answers"):
                    failed_questions.append(result["question"])
            
            # Simplified next round question building (can be extended as needed)
            next_round_questions = current_questions + failed_questions
            
            round_results = {
                "round": round_num,
                "input_questions": current_questions,
                "failed_questions": failed_questions,
                "next_round_questions": next_round_questions,
                "statistics": {
                    "input_count": len(current_questions),
                    "failed_count": len(failed_questions),
                    "next_round_count": len(next_round_questions)
                }
            }
            
            self.state_manager.save_round_data(round_num, "round_results", round_results)
            
            print(f"📊 Next round preparation completed: {len(next_round_questions)} questions")
            
        except Exception as e:
            logger.error(f"Failed to prepare next round: {e}", exc_info=True)
    
    def _get_solver_path(self, round_num: int) -> str:
        """Get Solver model path"""
        if round_num <= 2:
            # First 2 rounds use original model (no training)
            return self.config["solver_model_path"]
        else:
            # From round 3, use previous round's training output
            return self.modules["trl_trainer"].get_model_path_for_round(round_num)
    
    def _get_round_questions(self, round_num: int) -> list:
        """Get round questions"""
        if round_num == 1:
            return QuestionManager.load_questions_from_file(self.config["questions_file"])
        else:
            prev_results = self.state_manager.load_round_data(round_num - 1, "round_results")
            if prev_results:
                return prev_results.get("next_round_questions", [])
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        current_round = self.state_manager.get_current_round()
        completed_rounds = self.state_manager.get_completed_rounds()
        workspace_summary = self.state_manager.get_workspace_summary()
        
        return {
            "current_round": current_round,
            "completed_rounds": completed_rounds,
            "max_rounds": self.config["max_rounds"],
            "can_continue": current_round <= self.config["max_rounds"],
            "training_framework": "TRL_DPO",
            "workspace_summary": workspace_summary
        }
    
    def interactive_run(self):
        """Interactive run"""
        print("\n🎯 ========== Semi-Automated TRL Training Mode ==========")
        print("Using TRL DPO instead of VERL PPO training")
        print("Manual confirmation required after each round completion")
        print("Enter 'q' to quit, 's' for status, 'c' to continue to next round")
        
        while True:
            status = self.get_status()
            current_round = status["current_round"]
            
            print(f"\n📊 Current status:")
            print(f"   - Training framework: {status['training_framework']}")
            print(f"   - Current round: {current_round}")
            print(f"   - Completed rounds: {status['completed_rounds']}")
            print(f"   - Total rounds: {status['max_rounds']}")
            print(f"   - Can continue: {'Yes' if status['can_continue'] else 'No'}")
            
            if not status["can_continue"]:
                print("🎉 All rounds completed!")
                break
            
            print(f"\nPreparing to execute round {current_round} TRL training")
            user_input = input("Please select operation [c]continue [s]status [q]quit: ").strip().lower()
            
            if user_input == 'q':
                print("👋 Exiting TRL training")
                break
            elif user_input == 's':
                self._show_detailed_status()
                continue
            elif user_input == 'c':
                print(f"🚀 Starting round {current_round} TRL training...")
                result = self.execute_round(current_round)
                
                if result["success"]:
                    print(f"✅ Round {current_round} TRL training completed")
                    if "total_questions" in result:
                        print(f"📊 Processed {result['total_questions']} questions")
                else:
                    print(f"❌ Round {current_round} failed: {result.get('message', 'Unknown error')}")
                    
                    if result.get("can_retry", False):
                        retry_input = input("Please select operation [r]retry current round [s]status [q]quit: ").strip().lower()
                        
                        if retry_input == 'r':
                            print(f"🔄 Retrying round {current_round} TRL training...")
                            continue
                        elif retry_input == 's':
                            self._show_detailed_status()
                            continue
                        else:
                            print("👋 Exiting training")
                            break
                    else:
                        print(f"❌ This round cannot be retried, please check error cause")
                        break
            else:
                print("❓ Invalid input, please select again")
    
    def _show_detailed_status(self):
        """Show detailed status"""
        workspace_summary = self.state_manager.get_workspace_summary()
        
        print(f"\n📋 Detailed status:")
        print(f"   - Training framework: TRL DPO")
        print(f"   - Workspace: {workspace_summary['workspace_dir']}")
        print(f"   - Data files count: {workspace_summary['total_data_files']}")
        print(f"   - State file: {'Exists' if workspace_summary['state_exists'] else 'Not found'}")
        print(f"   - Config file: {'Exists' if workspace_summary['config_exists'] else 'Not found'}")
        print(f"   - Last updated: {workspace_summary['last_updated']}")
        
        # Show detailed status for each round
        current_round = self.state_manager.get_current_round()
        max_rounds = self.config["max_rounds"]
        
        print(f"\n📊 Round detailed status:")
        for round_num in range(1, min(current_round + 2, max_rounds + 1)):
            round_status = self.state_manager.get_round_status(round_num)
            status_emoji = "✅" if round_status["fully_completed"] else (
                "🔄" if round_status["status"] == "in_progress" else "⏸️"
            )
            print(f"   {status_emoji} Round {round_num}: {round_status['status']}")
            if round_status["completed_stages"]:
                print(f"      Completed: {', '.join(round_status['completed_stages'])}")
            if round_status["next_stage"]:
                print(f"      Next step: {round_status['next_stage']}")

def main():
    """Main function"""
    trainer = SemiAutoTRLTrainer()
    
    try:
        if not trainer.initialize():
            print("❌ TRL training system initialization failed")
            return
        
        trainer.interactive_run()
        
    except KeyboardInterrupt:
        print("\n🛑 User interrupted")
    except Exception as e:
        logger.error(f"TRL training error: {e}", exc_info=True)
        print(f"❌ TRL training error: {e}")

if __name__ == "__main__":
    main()