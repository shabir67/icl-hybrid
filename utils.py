"""
Utility functions for the example selection and inference pipeline.
"""

import os
import json
import pickle
import pandas as pd
from typing import List, Dict, Any, Optional
from datasets import load_dataset as hf_load_dataset
import numpy as np

def load_mmlu_data() -> pd.DataFrame:
    """Load MMLU dataset from HuggingFace."""
    splits = {
        'test': 'all/test-00000-of-00001.parquet', 
        'validation': 'all/validation-00000-of-00001.parquet', 
        'dev': 'all/dev-00000-of-00001.parquet', 
        'auxiliary_train': 'all/auxiliary_train-00000-of-00001.parquet'
    }
    df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])
    print(f"Loaded MMLU: {len(df)} samples")
    return df

def load_bb_data() -> pd.DataFrame:
    """Load BigBench dataset from HuggingFace."""
    splits = {
        'train': 'anachronisms/train-00000-of-00001.parquet', 
        'validation': 'anachronisms/validation-00000-of-00001.parquet'
    }
    df = pd.read_parquet("hf://datasets/tasksource/bigbench/" + splits["train"])
    print(f"Loaded BigBench: {len(df)} samples")
    return df

def load_gsm8k_data() -> pd.DataFrame:
    """Load GSM8K dataset from HuggingFace."""
    splits = {
        'train': 'main/train-00000-of-00001.parquet', 
        'test': 'main/test-00000-of-00001.parquet'
    }
    df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])
    print(f"Loaded GSM8K: {len(df)} samples")
    return df

def load_sst2_data() -> pd.DataFrame:
    """Load SST2 dataset from HuggingFace."""
    ds_sst2 = hf_load_dataset("SetFit/sst2")
    # Convert to DataFrame, handling multiple splits
    if hasattr(ds_sst2, 'keys'):
        # Concatenate all splits into one DataFrame
        dfs = []
        for split in ds_sst2.keys():
            split_df = ds_sst2[split].to_pandas()
            split_df['split'] = split
            dfs.append(split_df)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = ds_sst2.to_pandas()
    print(f"Loaded SST2: {len(df)} samples")
    return df

def load_sst5_data() -> pd.DataFrame:
    """Load SST5 dataset from HuggingFace."""
    ds_sst5 = hf_load_dataset("SetFit/sst5")
    # Convert to DataFrame, handling multiple splits
    if hasattr(ds_sst5, 'keys'):
        # Concatenate all splits into one DataFrame
        dfs = []
        for split in ds_sst5.keys():
            split_df = ds_sst5[split].to_pandas()
            split_df['split'] = split
            dfs.append(split_df)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = ds_sst5.to_pandas()
    print(f"Loaded SST5: {len(df)} samples")
    return df

def save_json(data: Any, file_path: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: str) -> Any:
    """Load data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_pickle(data: Any, file_path: str):
    """Save data to pickle file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path: str) -> Any:
    """Load data from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def validate_dataset_format(df: pd.DataFrame, dataset_name: str) -> bool:
    """Validate that dataset has expected columns."""
    expected_columns = {
        'mmlu': ['question', 'choices', 'answer', 'subject'],
        'bb': ['inputs', 'multiple_choice_targets'],
        'gsm8k': ['question', 'answer'],
        'sst2': ['text', 'label', 'label_text'],
        'sst5': ['text', 'label', 'label_text']
    }
    
    if dataset_name not in expected_columns:
        print(f"Unknown dataset: {dataset_name}")
        return False
    
    required_cols = expected_columns[dataset_name]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing columns for {dataset_name}: {missing_cols}")
        return False
    
    print(f"Dataset {dataset_name} format validation passed")
    return True

def print_dataset_info(df: pd.DataFrame, dataset_name: str):
    """Print basic information about a dataset."""
    print(f"\n{dataset_name.upper()} Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Print sample data
    print(f"Sample data:")
    if dataset_name == 'mmlu':
        print(f"Question: {df.iloc[0]['question'][:100]}...")
        print(f"Choices: {df.iloc[0]['choices']}")
        print(f"Answer: {df.iloc[0]['answer']}")
        print(f"Subject: {df.iloc[0]['subject']}")
    elif dataset_name == 'bb':
        print(f"Input: {df.iloc[0]['inputs'][:100]}...")
        print(f"Target: {df.iloc[0]['multiple_choice_targets']}")
    elif dataset_name == 'gsm8k':
        print(f"Question: {df.iloc[0]['question'][:100]}...")
        print(f"Answer: {df.iloc[0]['answer'][:100]}...")
    elif dataset_name in ['sst2', 'sst5']:
        print(f"Text: {df.iloc[0]['text'][:100]}...")
        print(f"Label: {df.iloc[0]['label']}")
        print(f"Label text: {df.iloc[0]['label_text']}")
    
    print("-" * 50)

def check_gpu_availability():
    """Check if GPU is available for model inference."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("GPU not available, using CPU")
            return False
    except ImportError:
        print("PyTorch not available")
        return False

def estimate_memory_usage(model_name: str) -> float:
    """Estimate memory usage for a model (in GB)."""
    memory_estimates = {
        'qwen0.6b': 2.0,
        'qwen1.8b': 4.0,
        'qwen3b': 7.0,
        'qwen7b': 15.0
    }
    return memory_estimates.get(model_name, 8.0)  # Default estimate

def create_experiment_config(
    datasets: List[str],
    methods: List[str],
    models: List[str],
    k: int = 16,
    n_iters: int = 3,
    demo_size: int = 1000,
    test_size: int = 100
) -> Dict:
    """Create experiment configuration."""
    config = {
        'datasets': datasets,
        'methods': methods,
        'models': models,
        'parameters': {
            'k': k,
            'n_iters': n_iters,
            'demo_size': demo_size,
            'test_size': test_size
        },
        'directories': {
            'selected_examples': 'selected_examples',
            'results': 'results',
            'plots': 'plots'
        }
    }
    return config

def print_experiment_summary(config: Dict):
    """Print experiment configuration summary."""
    print("Experiment Configuration:")
    print(f"Datasets: {', '.join(config['datasets'])}")
    print(f"Methods: {', '.join(config['methods'])}")
    print(f"Models: {', '.join(config['models'])}")
    print(f"Parameters: K={config['parameters']['k']}, "
          f"N_iters={config['parameters']['n_iters']}, "
          f"Demo_size={config['parameters']['demo_size']}, "
          f"Test_size={config['parameters']['test_size']}")
    
    total_experiments = len(config['datasets']) * len(config['methods']) * len(config['models'])
    print(f"Total experiments: {total_experiments}")

def clean_text(text: str) -> str:
    """Clean text for better processing."""
    if not isinstance(text, str):
        return str(text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might interfere
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    return text.strip()

def truncate_text(text: str, max_length: int = 512) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def get_dataset_stats(df: pd.DataFrame, dataset_name: str) -> Dict:
    """Get basic statistics about a dataset."""
    stats = {
        'total_samples': len(df),
        'columns': list(df.columns)
    }
    
    if dataset_name == 'mmlu':
        stats['subjects'] = df['subject'].nunique()
        stats['unique_subjects'] = df['subject'].unique().tolist()
        stats['answer_distribution'] = df['answer'].value_counts().to_dict()
    elif dataset_name in ['sst2', 'sst5']:
        stats['label_distribution'] = df['label_text'].value_counts().to_dict()
    elif dataset_name == 'bb':
        if 'multiple_choice_targets' in df.columns:
            # Flatten the list column to get distribution
            all_targets = []
            for targets in df['multiple_choice_targets']:
                all_targets.extend(targets)
            from collections import Counter
            stats['target_distribution'] = dict(Counter(all_targets))
    
    return stats

def validate_selection_file(file_path: str) -> bool:
    """Validate a selection file has required fields."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        required_fields = ['method', 'dataset', 'templates', 'test_examples']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"Selection file {file_path} missing fields: {missing_fields}")
            return False
        
        print(f"Selection file {file_path} validation passed")
        return True
    except Exception as e:
        print(f"Error validating selection file {file_path}: {e}")
        return False

def list_available_files(directory: str, extension: str = None) -> List[str]:
    """List available files in a directory."""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return []
    
    files = []
    for filename in os.listdir(directory):
        if extension is None or filename.endswith(extension):
            files.append(os.path.join(directory, filename))
    
    return sorted(files)

def setup_directories(directories: List[str]):
    """Create necessary directories."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/verified: {directory}")

if __name__ == "__main__":
    # Example usage and testing
    print("Testing utility functions...")
    
    # Test GPU availability
    check_gpu_availability()
    
    # Test dataset loading (comment out if datasets not available)
    try:
        df_sst2 = load_sst2_data()
        print_dataset_info(df_sst2, 'sst2')
        stats = get_dataset_stats(df_sst2, 'sst2')
        print("Dataset stats:", stats)
    except Exception as e:
        print(f"Dataset loading test failed: {e}")
    
    # Test directory setup
    setup_directories(['test_dir', 'test_dir/subdir'])
    
    print("Utility functions test completed!")