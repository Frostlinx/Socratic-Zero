#!/usr/bin/env python3
"""
Automated TRL Training Script
Fully automated training with retry mechanism and checkpoint recovery
"""

import os
import sys
import time
import logging
import datetime
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List

# Setup project paths
project_root = Path(__file__).parent.parent.absolute()
prosetting_root = project_root / "ProSetting"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(prosetting_root) not in sys.path:
    sys.path.insert(0, str(prosetting_root))

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
log_file_path = log_dir_path / 'auto_trainer.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file_path), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class AutoTRLTrainer:
    """Automated TRL trainer with full automation and retry mechanism"""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.round_controller = None
        self.modules = {}
        self.config = {}
        self.should_stop = False
        self.max_retries = 3
        self.retry_delay = 60  # Retry delay in seconds
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Signal handler for graceful shutdown"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True
        
    def initialize(self):
        """Initialize training environment"""
        logger.info("🚀 ========== Automated TRL Training System Initialization ==========")
        
        try:
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
            
            logger.info(f"✅ Automated TRL Training System Initialization Completed")
            logger.info(f"📊 Configuration Info:")
            logger.info(f"   - Total rounds: {self.config['max_rounds']}")
            logger.info(f"   - Save rounds: {self.config['save_rounds']}")
            logger.info(f"   - Training framework: TRL DPO")
            logger.info(f"   - Workspace: {self.state_manager.workspace_dir}")
            logger.info(f"   - Max retries: {self.max_retries}")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False
    
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
        
        # Automation specific configuration
        auto_retry_enabled = os.getenv("AUTO_RETRY_ENABLED", "true").lower() == "true"
        auto_continue_on_failure = os.getenv("AUTO_CONTINUE_ON_FAILURE", "false").lower() == "true"
        checkpoint_interval = int(os.getenv("CHECKPOINT_INTERVAL", "1"))  # Save checkpoint every round
        
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
            "training_framework": "TRL_DPO",
            "auto_retry_enabled": auto_retry_enabled,
            "auto_continue_on_failure": auto_continue_on_failure,
            "checkpoint_interval": checkpoint_interval
        }
        
        logger.info(f"🔍 Auto TRL Configuration loaded:")
        logger.info(f"   - Training framework: TRL DPO (Auto)")
        logger.info(f"   - Solver model: {solver_model_path}")
        logger.info(f"   - Max rounds: {max_rounds}")
        logger.info(f"   - Auto retry: {auto_retry_enabled}")
        logger.info(f"   - Continue on failure: {auto_continue_on_failure}")
        
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
    
    def run_full_training(self) -> Dict[str, Any]:
        """Run complete automated training workflow"""
        logger.info("🚀 ========== Starting Automated TRL Training ==========")
        
        training_start_time = datetime.datetime.now()
        total_rounds_completed = 0
        total_rounds_failed = 0
        training_summary = {
            "start_time": training_start_time.isoformat(),
            "rounds_completed": 0,
            "rounds_failed": 0,
            "total_questions_processed": 0,
            "training_errors": [],
            "final_status": "unknown"
        }
        
        try:
            current_round = self.state_manager.get_current_round()
            max_rounds = self.config["max_rounds"]
            
            logger.info(f"📊 Training plan: Start from round {current_round}, total {max_rounds} rounds")
            
            while current_round <= max_rounds and not self.should_stop:
                logger.info(f"\n🎯 ========== Auto Executing Round {current_round} Training ==========")
                
                round_result = self._execute_round_with_retry(current_round)
                
                if round_result["success"]:
                    total_rounds_completed += 1
                    training_summary["rounds_completed"] = total_rounds_completed
                    
                    if "total_questions" in round_result:
                        training_summary["total_questions_processed"] += round_result["total_questions"]
                    
                    logger.info(f"✅ Round {current_round} training completed successfully")
                    
                    # Check if checkpoint save is needed
                    if current_round % self.config["checkpoint_interval"] == 0:
                        self._save_training_checkpoint(current_round)
                    
                else:
                    total_rounds_failed += 1
                    training_summary["rounds_failed"] = total_rounds_failed
                    training_summary["training_errors"].append({
                        "round": current_round,
                        "error": round_result.get("message", "Unknown error"),
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    
                    logger.error(f"❌ Round {current_round} training failed: {round_result.get('message', 'Unknown error')}")
                    
                    # Decide whether to continue based on configuration
                    if not self.config.get("auto_continue_on_failure", False):
                        logger.error("🛑 Training stopped due to failure (auto_continue_on_failure=False)")
                        training_summary["final_status"] = "stopped_on_failure"
                        break
                    else:
                        logger.warning("⚠️ Skip failed round, continue to next round")
                
                # Move to next round
                current_round += 1
                
                # Rest between rounds
                if current_round <= max_rounds and not self.should_stop:
                    logger.info(f"⏸️ Rest 30 seconds between rounds...")
                    time.sleep(30)
            
            # Determine final status
            if self.should_stop:
                training_summary["final_status"] = "interrupted"
                logger.info("🛑 Training interrupted by user")
            elif current_round > max_rounds:
                training_summary["final_status"] = "completed"
                logger.info("🎉 All rounds training completed")
            
            # Calculate training duration
            training_end_time = datetime.datetime.now()
            training_duration = training_end_time - training_start_time
            training_summary["end_time"] = training_end_time.isoformat()
            training_summary["duration_seconds"] = training_duration.total_seconds()
            training_summary["duration_formatted"] = str(training_duration)
            
            # Save training summary
            self._save_training_summary(training_summary)
            
            # Print final report
            self._print_final_report(training_summary)
            
            return training_summary
            
        except Exception as e:
            logger.error(f"Automated training encountered serious error: {e}", exc_info=True)
            training_summary["final_status"] = "error"
            training_summary["fatal_error"] = str(e)
            return training_summary
    
    def _execute_round_with_retry(self, round_num: int) -> Dict[str, Any]:
        """Execute round with retry mechanism"""
        if not self.config.get("auto_retry_enabled", True):
            return self._execute_round(round_num)
        
        last_result = None
        for attempt in range(1, self.max_retries + 1):
            if self.should_stop:
                return {"success": False, "message": "Training interrupted"}
            
            logger.info(f"🔄 Round {round_num} training, attempt {attempt}/{self.max_retries}")
            
            result = self._execute_round(round_num)
            
            if result["success"]:
                if attempt > 1:
                    logger.info(f"✅ Round {round_num} succeeded after attempt {attempt}")
                return result
            
            last_result = result
            
            if attempt < self.max_retries:
                logger.warning(f"⚠️ Round {round_num} attempt {attempt} failed: {result.get('message', 'Unknown error')}")
                logger.info(f"⏳ Waiting {self.retry_delay} seconds before retry...")
                
                # Clean up GPU memory and resources
                try:
                    GPUManager.cleanup_and_release_models()
                    if hasattr(self.modules["trajectory_collector"], '_model_loaded'):
                        self.modules["trajectory_collector"].release_model()
                except Exception as cleanup_error:
                    logger.warning(f"Error during resource cleanup: {cleanup_error}")
                
                time.sleep(self.retry_delay)
            else:
                logger.error(f"❌ Round {round_num} failed after {self.max_retries} attempts")
        
        return last_result or {"success": False, "message": "All retry attempts failed"}
    
    def _execute_round(self, round_num: int) -> Dict[str, Any]:
        """Execute single TRL training round - same logic as semi-automated version"""
        logger.info(f"🚀 Executing round {round_num} TRL training")
        
        # Check round status
        round_status = self.state_manager.get_round_status(round_num)
        
        if round_status["fully_completed"]:
            logger.info(f"⚠️ Round {round_num} already completed, skipping execution")
            return {"success": True, "skipped": True, "message": "Round already completed"}
        
        try:
            # Get all questions
            all_questions = self._get_round_questions(round_num)
            if not all_questions:
                return {"success": False, "message": "Failed to load questions"}
            
            logger.info(f"📊 Total {len(all_questions)} questions to process")
            
            # Stage 1: Data collection
            if not self.state_manager.is_stage_completed(round_num, "data_collection"):
                logger.info("🚀 Stage 1: Data collection")
                collection_result = self._execute_data_collection(round_num, all_questions)
                if not collection_result["success"]:
                    return collection_result
                
                self.state_manager.save_round_progress(round_num, "data_collection", {
                    "trajectories": collection_result["data"]["all_trajectories"],
                    "questions": all_questions,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            else:
                logger.info("✅ Stage 1: Data collection completed, loading from checkpoint")
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
                logger.info("🚀 Stage 2: Data judging")
                judging_result = self._execute_data_judging(round_num, collection_result["data"]["all_trajectories"])
                if not judging_result["success"]:
                    return judging_result
                
                self.state_manager.save_round_progress(round_num, "data_judging", {
                    "judge_results": judging_result["data"]["judge_results"],
                    "timestamp": datetime.datetime.now().isoformat()
                })
            else:
                logger.info("✅ Stage 2: Data judging completed, loading from checkpoint")
                judging_data = self.state_manager.load_stage_data(round_num, "data_judging")
                judging_result = {
                    "success": True,
                    "data": {"judge_results": judging_data["judge_results"]}
                }
            
            # Stage 3: DPO data conversion
            if not self.state_manager.is_stage_completed(round_num, "dpo_conversion"):
                logger.info("🚀 Stage 3: DPO data conversion")
                conversion_result = self._execute_dpo_conversion(round_num, judging_result["data"]["judge_results"])
                if not conversion_result["success"]:
                    return conversion_result
                
                self.state_manager.save_round_progress(round_num, "dpo_conversion", {
                    "dataset_dir": conversion_result["data"]["dataset_dir"],
                    "timestamp": datetime.datetime.now().isoformat()
                })
            else:
                logger.info("✅ Stage 3: DPO data conversion completed, loading from checkpoint")
                conversion_data = self.state_manager.load_stage_data(round_num, "dpo_conversion")
                conversion_result = {
                    "success": True,
                    "data": {"dataset_dir": conversion_data["dataset_dir"]}
                }
            
            # Stage 4: TRL training - skip first 2 rounds, start from round 3
            if round_num <= 2:
                logger.info(f"⏭️ Stage 4: Skip TRL training (first 2 rounds, current: {round_num})")
                training_result = {"success": True, "skipped": True}
                self.state_manager.save_round_progress(round_num, "trl_training", {
                    "training_success": True,
                    "skipped": True,
                    "reason": f"Round {round_num} <= 2, no training in first 2 rounds",
                    "timestamp": datetime.datetime.now().isoformat()
                })
            else:
                if not self.state_manager.is_stage_completed(round_num, "trl_training"):
                    logger.info(f"🚀 Stage 4: TRL training (round {round_num} >= 3, start training)")
                    training_result = self._execute_trl_training(round_num, conversion_result["data"]["dataset_dir"])
                    
                    self.state_manager.save_round_progress(round_num, "trl_training", {
                        "training_success": training_result.get("success", False),
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                else:
                    logger.info("✅ Stage 4: TRL training completed")
                    training_result = {"success": True}
            
            # Stage 5: Prepare next round
            if not self.state_manager.is_stage_completed(round_num, "next_round_prep"):
                logger.info("🚀 Stage 5: Prepare next round")
                self._prepare_next_round(round_num, all_questions, judging_result["data"]["judge_results"])
                
                self.state_manager.save_round_progress(round_num, "next_round_prep", {
                    "preparation_complete": True,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            else:
                logger.info("✅ Stage 5: Next round preparation completed")
            
            # Check if training was successful
            training_success = training_result.get("success", False)
            
            if not training_success and not training_result.get("skipped", False):
                logger.warning(f"⚠️ Round {round_num} TRL training failed")
                return {
                    "success": False,
                    "round_num": round_num,
                    "message": f"Round {round_num} TRL training failed",
                    "can_retry": True
                }
            
            # Mark round as completed
            self.state_manager.mark_round_completed(round_num, True)
            
            logger.info(f"✅ Round {round_num} TRL training completed successfully")
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
            
            logger.info(f"✅ Data collection completed, collected {len(normalized_trajectories)} trajectories")
            
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
            logger.info("📊 Executing judging calculation...")
            
            rewards, judge_results = self.modules["reward_calculator"].compute_solver_reward_local(trajectories)
            
            # Save judge results to file
            processor = self.modules["solver_data_processor"]
            judge_file = processor.save_judge_results(judge_results, round_num)
            
            logger.info(f"✅ Judging completed, average reward: {sum(rewards)/len(rewards):.3f}")
            logger.info(f"💾 Judge results saved: {judge_file}")
            
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
            logger.info("🔄 Converting judge results to DPO format...")
            
            converter = self.modules["dpo_converter"]
            dataset_dir = converter.create_dataset_for_round(round_num, judge_results)
            
            # Get dataset statistics
            stats = converter.get_dataset_stats(dataset_dir)
            
            logger.info(f"✅ DPO data conversion completed")
            logger.info(f"📊 Training samples: {stats['files']['train']['samples']}")
            logger.info(f"📊 Validation samples: {stats['files']['validation']['samples']}")
            logger.info(f"💾 Dataset directory: {dataset_dir}")
            
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
            logger.info(f"🚀 Starting TRL DPO training...")
            
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
                logger.info(f"🎉 TRL training completed successfully")
            else:
                logger.error(f"❌ TRL training failed")
            
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
            
            logger.info(f"📊 Next round preparation completed: {len(next_round_questions)} questions")
            
        except Exception as e:
            logger.error(f"Failed to prepare next round: {e}", exc_info=True)
    
    def _get_solver_path(self, round_num: int) -> str:
        """Get Solver model path"""
        if round_num == 1:
            return self.config["solver_model_path"]
        else:
            # Use TRL trainer's path retrieval logic
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
    
    def _save_training_checkpoint(self, round_num: int):
        """Save training checkpoint"""
        try:
            checkpoint_data = {
                "round_num": round_num,
                "timestamp": datetime.datetime.now().isoformat(),
                "config": self.config,
                "status": self.state_manager.get_workspace_summary()
            }
            
            checkpoint_file = self.state_manager.workspace_dir / f"checkpoint_round_{round_num}.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Training checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _save_training_summary(self, summary: Dict[str, Any]):
        """Save training summary"""
        try:
            summary_file = self.state_manager.workspace_dir / "auto_training_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 Training summary saved: {summary_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save training summary: {e}")
    
    def _print_final_report(self, summary: Dict[str, Any]):
        """Print final report"""
        logger.info("\n" + "="*60)
        logger.info("🎯 Fully Automated TRL Training Final Report")
        logger.info("="*60)
        logger.info(f"📅 Start time: {summary['start_time']}")
        logger.info(f"📅 End time: {summary.get('end_time', 'N/A')}")
        logger.info(f"⏱️ Training duration: {summary.get('duration_formatted', 'N/A')}")
        logger.info(f"✅ Successful rounds: {summary['rounds_completed']}")
        logger.info(f"❌ Failed rounds: {summary['rounds_failed']}")
        logger.info(f"📊 Questions processed: {summary['total_questions_processed']}")
        logger.info(f"🏁 Final status: {summary['final_status']}")
        
        if summary['training_errors']:
            logger.info(f"\n⚠️ Training errors ({len(summary['training_errors'])} items):")
            for error in summary['training_errors']:
                logger.info(f"   - Round {error['round']}: {error['error']}")
        
        logger.info("="*60)

def main():
    """Main function"""
    trainer = AutoTRLTrainer()
    
    try:
        if not trainer.initialize():
            logger.error("❌ Fully automated TRL training system initialization failed")
            return 1
        
        # Run complete training
        result = trainer.run_full_training()
        
        # Return exit code based on results
        if result["final_status"] == "completed":
            logger.info("🎉 Fully automated training completed successfully")
            return 0
        elif result["final_status"] == "interrupted":
            logger.info("🛑 Training interrupted by user")
            return 130  # SIGINT exit code
        else:
            logger.error("❌ Training failed to complete successfully")
            return 1
        
    except KeyboardInterrupt:
        logger.info("\n🛑 User interrupted training")
        return 130
    except Exception as e:
        logger.error(f"Fully automated TRL training error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)