#!/usr/bin/env python3
"""
Mean@K Evaluation Script

Simple evaluation script for calculating Mean@32 and other metrics
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup project paths
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Note: This import needs to be implemented based on your evaluation module structure
# from src.evaluation import MathEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate math problem solving with Mean@K metric')
    parser.add_argument('--predictions', '-p', required=True, 
                       help='Path to predictions JSON file (trajectories)')
    parser.add_argument('--ground-truth', '-g', required=True,
                       help='Path to ground truth JSON file')
    parser.add_argument('--k', type=int, default=32,
                       help='K value for Mean@K calculation (default: 32)')
    parser.add_argument('--output', '-o', 
                       help='Path to save evaluation results (optional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input files
    predictions_file = Path(args.predictions)
    ground_truth_file = Path(args.ground_truth)
    
    if not predictions_file.exists():
        logger.error(f"Predictions file not found: {predictions_file}")
        return 1
    
    if not ground_truth_file.exists():
        logger.error(f"Ground truth file not found: {ground_truth_file}")
        return 1
    
    logger.info(f"🚀 Starting Mean@{args.k} evaluation")
    logger.info(f"📁 Predictions: {predictions_file}")
    logger.info(f"📁 Ground Truth: {ground_truth_file}")
    
    try:
        # Initialize evaluator
        evaluator = MathEvaluator()
        
        # Run evaluation
        results = evaluator.evaluate_files(
            predictions_file=predictions_file,
            ground_truth_file=ground_truth_file,
            k=args.k
        )
        
        # Print results
        print("\n" + "="*50)
        print("📊 EVALUATION RESULTS")
        print("="*50)
        print(f"Mean@{args.k:2d}:     {results['mean_at_k']:.4f}")
        print(f"Accuracy:     {results['accuracy']:.4f}")
        print(f"Total Qs:     {results['total_questions']}")
        print(f"Matched Qs:   {results['matched_questions']}")
        print(f"Match Rate:   {results['matched_questions']/results['total_questions']:.4f}")
        print("="*50)
        
        # Save results if output path provided
        if args.output:
            output_file = Path(args.output)
            evaluator.save_results(output_file)
            print(f"💾 Results saved to: {output_file}")
        
        logger.info("✅ Evaluation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)