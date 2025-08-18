"""
Main runner script for the example selection and evaluation pipeline.
"""

import argparse
import os
import sys
from typing import List
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from example_selection import run_example_selection
from inference_evaluation import run_full_evaluation, create_results_summary
from utils import (
    create_experiment_config, 
    print_experiment_summary, 
    setup_directories,
    check_gpu_availability,
    estimate_memory_usage
)

def run_example_selection_pipeline(datasets: List[str], k: int = 16,
                                  output_dir: str = "selected_examples"):
    """Run example selection for specified datasets with automatic sizing."""
    print("="*60)
    print("RUNNING EXAMPLE SELECTION PIPELINE")
    print("="*60)
    
    setup_directories([output_dir])
    
    for dataset in datasets:
        try:
            print(f"\nProcessing {dataset.upper()}...")
            run_example_selection(
                dataset_name=dataset,
                k=k,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            continue
    
    print(f"\nExample selection completed! Files saved to: {output_dir}")

def run_inference_pipeline(models: List[str], 
                          selected_examples_dir: str = "selected_examples",
                          results_dir: str = "results",
                          plots_dir: str = "plots"):
    """Run inference and evaluation pipeline."""
    print("="*60)
    print("RUNNING INFERENCE AND EVALUATION PIPELINE")
    print("="*60)
    
    setup_directories([results_dir, plots_dir])
    
    # Check GPU and memory requirements
    gpu_available = check_gpu_availability()
    
    for model in models:
        memory_req = estimate_memory_usage(model)
        print(f"Model {model} estimated memory requirement: {memory_req:.1f} GB")
    
    if not gpu_available:
        print("Warning: No GPU detected. Inference will be slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Run evaluation
    try:
        results = run_full_evaluation(
            selected_examples_dir=selected_examples_dir,
            models=models,
            output_dir=results_dir
        )
        
        print(f"Completed {len(results)} evaluations")
        
        # Create summary
        summary_df = create_results_summary(results_dir)
        
        if summary_df is not None:
            print("\n" + "="*60)
            print("RESULTS SUMMARY")
            print("="*60)
            print(summary_df.to_string(index=False))
        
    except Exception as e:
        print(f"Error in inference pipeline: {e}")

def main():
    parser = argparse.ArgumentParser(description="Example Selection for In-Context Learning")
    parser.add_argument('--mode', choices=['selection', 'inference', 'full'], 
                       default='full', help='Pipeline mode to run')
    parser.add_argument('--datasets', nargs='+', 
                       default=['mmlu', 'bb', 'gsm8k', 'sst2', 'sst5'],
                       help='Datasets to process')
    parser.add_argument('--models', nargs='+',
                       default=['qwen0.6b', 'qwen1.8b'],
                       help='Models to evaluate')
    parser.add_argument('--k', type=int, default=16,
                       help='Number of examples to select')
    parser.add_argument('--n-iters', type=int, default=3,
                       help='Number of iterations for iterative methods')
    parser.add_argument('--demo-size', type=int, default=1000,
                       help='Size of demonstration pool')
    parser.add_argument('--test-size', type=int, default=100,
                       help='Size of test set')
    parser.add_argument('--output-dir', default='selected_examples',
                       help='Output directory for selected examples')
    parser.add_argument('--results-dir', default='results',
                       help='Output directory for results')
    parser.add_argument('--plots-dir', default='plots',
                       help='Output directory for plots')
    parser.add_argument('--config-file', type=str,
                       help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config_file}")
    else:
        # Create config from arguments
        config = create_experiment_config(
            datasets=args.datasets,
            methods=['zero_shot', 'random_shot', 'similarity_baseline', 'hybrid_similarity_diversity'],
            models=args.models,
            k=args.k,
            n_iters=args.n_iters,
            demo_size=args.demo_size,
            test_size=args.test_size
        )
        
        # Update directories from args
        config['directories']['selected_examples'] = args.output_dir
        config['directories']['results'] = args.results_dir
        config['directories']['plots'] = args.plots_dir
    
    # Print experiment summary
    print_experiment_summary(config)
    
    # Run selected pipeline
    if args.mode in ['selection', 'full']:
        run_example_selection_pipeline(
            datasets=config['datasets'],
            k=config['parameters']['k'],
            output_dir=config['directories']['selected_examples']
        )
    
    if args.mode in ['inference', 'full']:
        run_inference_pipeline(
            models=config['models'],
            selected_examples_dir=config['directories']['selected_examples'],
            results_dir=config['directories']['results'],
            plots_dir=config['directories']['plots']
        )
    
    print("\nPipeline completed!")

def create_sample_config():
    """Create a sample configuration file for HPC usage."""
    config = create_experiment_config(
        datasets=['sst2', 'sst5'],  # Start with smaller datasets
        methods=['zero_shot', 'random_shot', 'similarity_baseline', 'hybrid_similarity_diversity'],
        models=['qwen0.6b', 'qwen1.8b'],  # HPC-appropriate models
        k=16,
        n_iters=2,
        demo_size=5000,  # Larger for full dataset usage
        test_size=1000
    )
    
    with open('sample_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration saved to: sample_config.json")
    print("You can modify this file and use it with --config-file option")

def quick_test():
    """Run a quick test with minimal data for HPC testing."""
    print("Running quick test for HPC...")
    
    # Test with minimal parameters
    config = create_experiment_config(
        datasets=['sst2'],  # Single dataset
        methods=['zero_shot', 'random_shot'],  # Two simple methods
        models=['qwen0.6b'],  # Smallest model
        k=8,
        n_iters=1,
        demo_size=500,
        test_size=100
    )
    
    print_experiment_summary(config)
    
    # Run example selection
    run_example_selection_pipeline(
        datasets=config['datasets'],
        k=config['parameters']['k'],
        output_dir='test_selected_examples'
    )
    
    print("Quick test completed! Check 'test_selected_examples' directory.")
    print("For full inference testing, submit an HPC job with GPU resources.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'create-config':
            create_sample_config()
        elif sys.argv[1] == 'quick-test':
            quick_test()
        else:
            main()
    else:
        main()