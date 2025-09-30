#!/usr/bin/env python3
"""
ProSetting Status Checker Tool
Used to check current system status, configuration and data integrity
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime

# Setup project paths
project_root = Path(__file__).parent.parent.parent.absolute()
prosetting_root = project_root / "ProSetting"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(prosetting_root) not in sys.path:
    sys.path.insert(0, str(prosetting_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProSettingStatusChecker:
    """ProSetting system status checker"""
    
    def __init__(self, workspace_dir: str = None):
        self.workspace_dir = workspace_dir or os.getenv("WORKSPACE_DIR", "/tmp/prosetting_workspace")
        self.status_report = {}
        
    def check_environment(self):
        """Check environment configuration"""
        print("🔧 ========== Environment Configuration Check ==========")
        
        env_status = {}
        
        # Check key environment variables
        key_env_vars = [
            "SOLVER_MODEL_PATH",
            "QUESTIONS_FILE", 
            "WORKSPACE_DIR",
            "TOTAL_ROUNDS",
            "SAVE_ROUNDS",
            "TEACHER_BASE_URL",
            "TRL_NUM_PROCESSES",
            "TRL_MIXED_PRECISION"
        ]
        
        for var in key_env_vars:
            value = os.getenv(var)
            if value:
                env_status[var] = {"value": value, "status": "✅"}
                print(f"  ✅ {var}: {value}")
            else:
                env_status[var] = {"value": None, "status": "❌"}
                print(f"  ❌ {var}: Not set")
        
        self.status_report["environment"] = env_status
        return env_status
    
    def check_file_paths(self):
        """Check file paths"""
        print("\n📁 ========== File Path Check ==========")
        
        file_status = {}
        
        # Check model path
        solver_model_path = os.getenv("SOLVER_MODEL_PATH")
        if solver_model_path:
            if Path(solver_model_path).exists():
                file_status["solver_model"] = {"path": solver_model_path, "exists": True, "status": "✅"}
                print(f"  ✅ Solver model: {solver_model_path}")
            else:
                file_status["solver_model"] = {"path": solver_model_path, "exists": False, "status": "❌"}
                print(f"  ❌ Solver model not found: {solver_model_path}")
        
        # Check questions file
        questions_file = os.getenv("QUESTIONS_FILE")
        if questions_file:
            if Path(questions_file).exists():
                try:
                    with open(questions_file, 'r', encoding='utf-8') as f:
                        questions_data = json.load(f)
                    question_count = len(questions_data)
                    file_status["questions_file"] = {
                        "path": questions_file, 
                        "exists": True, 
                        "count": question_count,
                        "status": "✅"
                    }
                    print(f"  ✅ Questions file: {questions_file} ({question_count} questions)")
                except Exception as e:
                    file_status["questions_file"] = {
                        "path": questions_file, 
                        "exists": True, 
                        "error": str(e),
                        "status": "⚠️"
                    }
                    print(f"  ⚠️ Questions file format error: {e}")
            else:
                file_status["questions_file"] = {"path": questions_file, "exists": False, "status": "❌"}
                print(f"  ❌ Questions file not found: {questions_file}")
        
        # Check workspace
        if Path(self.workspace_dir).exists():
            file_status["workspace"] = {"path": self.workspace_dir, "exists": True, "status": "✅"}
            print(f"  ✅ Workspace: {self.workspace_dir}")
        else:
            file_status["workspace"] = {"path": self.workspace_dir, "exists": False, "status": "❌"}
            print(f"  ❌ Workspace not found: {self.workspace_dir}")
        
        self.status_report["file_paths"] = file_status
        return file_status
    
    def check_training_state(self):
        """Check training state"""
        print("\n📊 ========== Training Status Check ==========")
        
        training_status = {}
        
        try:
            from core import StateManager
            
            state_manager = StateManager(self.workspace_dir)
            
            # Check training configuration
            config = state_manager.load_training_config()
            if config:
                training_status["config"] = {"exists": True, "data": config, "status": "✅"}
                print(f"  ✅ Training configuration loaded")
                print(f"    - Total rounds: {config.get('max_rounds', 'N/A')}")
                print(f"    - Save rounds: {config.get('save_rounds', 'N/A')}")
                print(f"    - Training framework: {config.get('training_framework', 'N/A')}")
            else:
                training_status["config"] = {"exists": False, "status": "❌"}
                print(f"  ❌ Training configuration not found")
            
            # Check training state
            state = state_manager.load_training_state()
            if state:
                current_round = state_manager.get_current_round()
                completed_rounds = state_manager.get_completed_rounds()
                
                training_status["state"] = {
                    "exists": True,
                    "current_round": current_round,
                    "completed_rounds": completed_rounds,
                    "data": state,
                    "status": "✅"
                }
                
                print(f"  ✅ Training state loaded")
                print(f"    - Current round: {current_round}")
                print(f"    - Completed rounds: {completed_rounds}")
                print(f"    - Last updated: {state.get('last_updated', 'N/A')}")
            else:
                training_status["state"] = {"exists": False, "status": "❌"}
                print(f"  ❌ Training state not found")
            
            # Check round detailed status
            if config:
                max_rounds = config.get("max_rounds", int(os.getenv("TOTAL_ROUNDS", "10")))
                round_details = {}
                
                for round_num in range(1, min(max_rounds + 1, 6)):  # Check at most 5 rounds
                    round_status = state_manager.get_round_status(round_num)
                    round_details[f"round_{round_num}"] = round_status
                    
                    status_emoji = "✅" if round_status["fully_completed"] else (
                        "🔄" if round_status["status"] == "in_progress" else "⏸️"
                    )
                    
                    print(f"    {status_emoji} Round {round_num}: {round_status['status']}")
                    if round_status["completed_stages"]:
                        print(f"      Completed: {', '.join(round_status['completed_stages'])}")
                    if round_status["next_stage"]:
                        print(f"      Next step: {round_status['next_stage']}")
                
                training_status["round_details"] = round_details
            
        except Exception as e:
            training_status["error"] = str(e)
            print(f"  ❌ Training status check failed: {e}")
        
        self.status_report["training_state"] = training_status
        return training_status
    
    def check_data_integrity(self):
        """Check data integrity"""
        print("\n💾 ========== Data Integrity Check ==========")
        
        data_status = {}
        
        workspace_path = Path(self.workspace_dir)
        
        if not workspace_path.exists():
            data_status["workspace_missing"] = True
            print(f"  ❌ Workspace not found: {self.workspace_dir}")
            self.status_report["data_integrity"] = data_status
            return data_status
        
        # Check data directory structure
        expected_dirs = ["data", "datasets", "training_data", "checkpoints", "logs", "results"]
        
        for dir_name in expected_dirs:
            dir_path = workspace_path / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*")))
                data_status[dir_name] = {"exists": True, "file_count": file_count, "status": "✅"}
                print(f"  ✅ {dir_name} directory: {file_count} files")
            else:
                data_status[dir_name] = {"exists": False, "status": "⚠️"}
                print(f"  ⚠️ {dir_name} directory not found")
        
        # Check round data files
        data_dir = workspace_path / "data"
        if data_dir.exists():
            round_files = {}
            
            for round_file in data_dir.glob("round_*_*.json"):
                round_files[round_file.name] = {
                    "path": str(round_file),
                    "size": round_file.stat().st_size,
                    "modified": datetime.datetime.fromtimestamp(round_file.stat().st_mtime).isoformat()
                }
                print(f"    📄 {round_file.name}: {round_file.stat().st_size} bytes")
            
            data_status["round_files"] = round_files
        
        # Check TRL datasets
        datasets_dir = workspace_path / "datasets"
        if datasets_dir.exists():
            dataset_rounds = {}
            
            for round_dir in datasets_dir.glob("round_*"):
                if round_dir.is_dir():
                    train_file = round_dir / "train.parquet"
                    val_file = round_dir / "validation.parquet"
                    info_file = round_dir / "dataset_info.json"
                    
                    dataset_rounds[round_dir.name] = {
                        "train_exists": train_file.exists(),
                        "validation_exists": val_file.exists(),
                        "info_exists": info_file.exists(),
                        "status": "✅" if all([train_file.exists(), val_file.exists()]) else "⚠️"
                    }
                    
                    status = "✅" if all([train_file.exists(), val_file.exists()]) else "⚠️"
                    print(f"    {status} {round_dir.name}: train={train_file.exists()}, val={val_file.exists()}")
            
            data_status["datasets"] = dataset_rounds
        
        self.status_report["data_integrity"] = data_status
        return data_status
    
    def check_module_imports(self):
        """Check module imports"""
        print("\n📦 ========== Module Import Check ==========")
        
        import_status = {}
        
        # Check core modules
        core_modules = [
            ("collectors", "Data collection modules"),
            ("processors", "Data processing modules"),
            ("datasets", "Dataset modules"),
            ("trainers", "Trainer modules"),
            ("managers", "Manager modules"),
            ("core", "Core modules")
        ]
        
        for module_name, description in core_modules:
            try:
                __import__(module_name)
                import_status[module_name] = {"status": "✅", "description": description}
                print(f"  ✅ {description}: {module_name}")
            except ImportError as e:
                import_status[module_name] = {"status": "❌", "error": str(e), "description": description}
                print(f"  ❌ {description}: {e}")
        
        # Check TRL related modules
        trl_modules = [
            ("trl", "TRL training framework"),
            ("accelerate", "Accelerate distributed training"),
            ("pandas", "Data processing"),
            ("pyarrow", "Parquet support")
        ]
        
        for module_name, description in trl_modules:
            try:
                __import__(module_name)
                import_status[module_name] = {"status": "✅", "description": description}
                print(f"  ✅ {description}: {module_name}")
            except ImportError as e:
                import_status[module_name] = {"status": "❌", "error": str(e), "description": description}
                print(f"  ❌ {description}: {e}")
        
        self.status_report["module_imports"] = import_status
        return import_status
    
    def generate_report(self):
        """Generate complete status report"""
        print("\n📋 ========== Generate Status Report ==========")
        
        # Run all checks
        self.check_environment()
        self.check_file_paths()
        self.check_training_state()
        self.check_data_integrity()
        self.check_module_imports()
        
        # Generate report summary
        report_summary = {
            "check_time": datetime.datetime.now().isoformat(),
            "workspace_dir": self.workspace_dir,
            "overall_status": self._calculate_overall_status(),
            "details": self.status_report
        }
        
        # Save report to file
        report_file = Path(self.workspace_dir) / "status_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_summary, f, ensure_ascii=False, indent=2)
        
        print(f"📄 Status report saved: {report_file}")
        
        # Print summary
        self._print_summary()
        
        return report_summary
    
    def _calculate_overall_status(self):
        """Calculate overall status"""
        total_checks = 0
        passed_checks = 0
        
        for category, details in self.status_report.items():
            if isinstance(details, dict):
                for item, status in details.items():
                    if isinstance(status, dict) and "status" in status:
                        total_checks += 1
                        if status["status"] == "✅":
                            passed_checks += 1
        
        if total_checks == 0:
            return {"status": "unknown", "score": 0}
        
        score = passed_checks / total_checks
        
        if score >= 0.9:
            status = "excellent"
        elif score >= 0.7:
            status = "good"
        elif score >= 0.5:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "status": status,
            "score": score,
            "passed": passed_checks,
            "total": total_checks
        }
    
    def _print_summary(self):
        """Print status summary"""
        overall = self._calculate_overall_status()
        
        print(f"\n🎯 ========== Status Summary ==========")
        print(f"Overall status: {overall['status'].upper()}")
        print(f"Check pass rate: {overall['passed']}/{overall['total']} ({overall['score']:.1%})")
        
        if overall['score'] >= 0.9:
            print("🎉 System status excellent, ready for training!")
        elif overall['score'] >= 0.7:
            print("✅ System status good, recommend checking warning items")
        elif overall['score'] >= 0.5:
            print("⚠️ System status fair, recommend fixing failed items before running")
        else:
            print("❌ System status poor, please fix critical issues before use")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ProSetting system status checker tool")
    parser.add_argument("--workspace", "-w", help="Specify workspace directory")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick check (skip detailed data checks)")
    
    args = parser.parse_args()
    
    checker = ProSettingStatusChecker(args.workspace)
    
    if args.quick:
        print("🚀 Quick status check...")
        checker.check_environment()
        checker.check_file_paths()
        checker.check_module_imports()
    else:
        print("🔍 Complete status check...")
        checker.generate_report()

if __name__ == "__main__":
    main()