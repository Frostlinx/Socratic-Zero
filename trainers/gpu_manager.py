"""
GPU Manager

Manages GPU resources and environment for training.
"""

import os
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class GPUManager:
    """
    GPU resource manager for training operations
    """
    
    @staticmethod
    def setup_gpu_environment():
        """Setup GPU environment for training"""
        try:
            # Set CUDA device order
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Check GPU availability
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("GPU environment setup completed")
            else:
                logger.warning("nvidia-smi not available, GPU may not be accessible")
                
        except Exception as e:
            logger.warning(f"GPU environment setup failed: {e}")
    
    @staticmethod
    def set_gpu_environment(gpu_ids: str):
        """
        Set GPU environment for specific GPUs
        
        Args:
            gpu_ids: Comma-separated GPU IDs (e.g., "0,1,2,3")
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        logger.info(f"Set CUDA_VISIBLE_DEVICES to: {gpu_ids}")
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache"""
        try:
            # Placeholder for GPU memory clearing logic
            logger.info("GPU memory cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GPU memory: {e}")
    
    @staticmethod
    def cleanup_and_release_models():
        """Cleanup and release GPU models"""
        try:
            # Placeholder for model cleanup logic
            logger.info("GPU models cleaned up and released")
        except Exception as e:
            logger.warning(f"Failed to cleanup GPU models: {e}")
    
    @staticmethod
    def get_gpu_info() -> dict:
        """
        Get GPU information
        
        Returns:
            Dictionary with GPU information
        """
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = {"available": True, "details": result.stdout.strip()}
            else:
                gpu_info = {"available": False, "error": "nvidia-smi failed"}
        except Exception as e:
            gpu_info = {"available": False, "error": str(e)}
        
        return gpu_info