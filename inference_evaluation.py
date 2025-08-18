"""
Inference and evaluation module for in-context learning experiments.
Loads selected examples and runs inference with Qwen models.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import Counter
import re

# Model configurations - Updated for HPC usage
MODEL_CONFIGS = {
    'qwen0.6b': {
        'model_name': 'Qwen/Qwen3-0.6b',
        'device_map': 'auto',
        'torch_dtype': "auto",
        'max_new_tokens': 32768,
        'max_length': 2048
    },
    'qwen1.7b': {
        'model_name': 'Qwen/Qwen3-1.7b',
        'device_map': 'auto',
        'torch_dtype': "auto",
        'max_new_tokens': 32768,
        'max_length': 2048
    },
    'qwen4b': {
        'model_name': 'Qwen/Qwen3-4B',
        'device_map': 'auto',
        'torch_dtype': "auto",
        'max_new_tokens': 32768,
        'max_length': 2048
    },
    'qwen8b': {
        'model_name': 'Qwen/Qwen3-8B',
        'device_map': 'auto',
        'torch_dtype': "auto",
        'max_new_tokens': 32768,
        'max_length': 2048
    },
    'qwen14b': {
        'model_name': 'Qwen/Qwen3-14B',
        'device_map': 'auto',
        'torch_dtype': "auto",
        'max_new_tokens': 32768,
        'max_length': 2048
    },
    'qwen32b': {
        'model_name': 'Qwen/Qwen3-32B',
        'device_map': 'auto',
        'torch_dtype': "auto",
        'max_new_tokens': 32768,
        'max_length': 2048
    }
}

def load_selection_file(file_path: str) -> Dict:
    """Load a selection file created by example_selection.py"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded selection file: {file_path}")
    print(f"Method: {data['method']}, Dataset: {data['dataset']}")
    return data

def load_model_and_tokenizer(model_config: Dict):
    """Load Qwen model and tokenizer - HPC optimized."""
    model_name = model_config['model_name']
    print(f"Loading model: {model_name}")
    
    # Check if we're on HPC with GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
            
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def get_default_templates(dataset_name: str) -> Dict:
    """Get default templates for each dataset if not provided."""
    templates = {
        'mmlu': {
            'zero_shot': "Question: {question}\nChoices: {choices}\nAnswer:",
            'few_shot': "Here are some examples:\n\n{examples}\n\nNow answer this question:\nQuestion: {question}\nChoices: {choices}\nAnswer:",
            'example_format': "Question: {question}\nChoices: {choices}\nAnswer: ({answer}) {answer_text}"
        },
        'bb': {
            'zero_shot': "Statement: {statement}\nIs this statement plausible? Answer Yes or No:",
            'few_shot': "Here are some examples:\n\n{examples}\n\nNow answer this:\nStatement: {statement}\nIs this statement plausible? Answer Yes or No:",
            'example_format': "Statement: {inputs}\nAnswer: {answer}"
        },
        'gsm8k': {
            'zero_shot': "Problem: {problem}\nSolution:",
            'few_shot': "Here are some examples:\n\n{examples}\n\nNow solve this problem:\nProblem: {problem}\nSolution:",
            'example_format': "Problem: {question}\nSolution: {answer}"
        },
        'sst2': {
            'zero_shot': "Text: {text}\nSentiment (positive or negative):",
            'few_shot': "Here are some examples:\n\n{examples}\n\nNow classify this text:\nText: {text}\nSentiment (positive or negative):",
            'example_format': "Text: {text}\nSentiment: {label_text}"
        },
        'sst5': {
            'zero_shot': "Text: {text}\nSentiment (very negative, negative, neutral, positive, very positive):",
            'few_shot': "Here are some examples:\n\n{examples}\n\nNow classify this text:\nText: {text}\nSentiment (very negative, negative, neutral, positive, very positive):",
            'example_format': "Text: {text}\nSentiment: {label_text}"
        }
    }
    return templates.get(dataset_name, {})

def format_single_example(example: Dict, dataset_name: str, template: str = None) -> str:
    """Format a single example based on dataset type."""
    if template is None:
        templates = get_default_templates(dataset_name)
        template = templates.get('example_format', '')
    
    try:
        if dataset_name == 'mmlu':
            choices_str = ', '.join([f"({i}) {choice}" for i, choice in enumerate(example['choices'])])
            answer_text = example['choices'][example['answer']]
            return template.format(
                question=example['question'],
                choices=choices_str,
                answer=example['answer'],
                answer_text=answer_text
            )
        elif dataset_name == 'bb':
            answer = "Yes" if example['multiple_choice_targets'][0] == "Yes" else "No"
            return template.format(
                inputs=example['inputs'],
                answer=answer
            )
        elif dataset_name == 'gsm8k':
            return template.format(
                question=example['question'],
                answer=example['answer']
            )
        elif dataset_name in ['sst2', 'sst5']:
            return template.format(
                text=example['text'],
                label_text=example['label_text']
            )
    except KeyError as e:
        print(f"Warning: Missing key {e} in example for dataset {dataset_name}")
        return f"Example formatting error for {dataset_name}"
    
    return "Unsupported dataset format"

def format_prompt_for_inference(test_example: Dict, selected_examples: List[Dict], 
                               templates: Dict, dataset_name: str) -> str:
    """Format the prompt for inference."""
    # Use default templates if not provided or incomplete
    if not templates or not all(key in templates for key in ['zero_shot', 'few_shot']):
        print(f"Warning: Using default templates for {dataset_name}")
        templates = get_default_templates(dataset_name)
    
    if not selected_examples:  # Zero-shot
        template = templates['zero_shot']
        try:
            if dataset_name == 'mmlu':
                choices_str = ', '.join([f"({i}) {choice}" for i, choice in enumerate(test_example['choices'])])
                return template.format(question=test_example['question'], choices=choices_str)
            elif dataset_name == 'bb':
                return template.format(statement=test_example['inputs'])
            elif dataset_name == 'gsm8k':
                return template.format(problem=test_example['question'])
            elif dataset_name in ['sst2', 'sst5']:
                return template.format(text=test_example['text'])
        except KeyError as e:
            print(f"Error formatting zero-shot prompt for {dataset_name}: {e}")
            return f"Error formatting prompt: missing {e}"
    else:  # Few-shot
        # Format examples
        example_strings = []
        example_template = templates.get('example_format', '')
        
        for example in selected_examples:
            formatted_example = format_single_example(example, dataset_name, example_template)
            example_strings.append(formatted_example)
        
        examples_text = '\n\n'.join(example_strings)
        template = templates['few_shot']
        
        try:
            if dataset_name == 'mmlu':
                choices_str = ', '.join([f"({i}) {choice}" for i, choice in enumerate(test_example['choices'])])
                return template.format(examples=examples_text, question=test_example['question'], choices=choices_str)
            elif dataset_name == 'bb':
                return template.format(examples=examples_text, statement=test_example['inputs'])
            elif dataset_name == 'gsm8k':
                return template.format(examples=examples_text, problem=test_example['question'])
            elif dataset_name in ['sst2', 'sst5']:
                return template.format(examples=examples_text, text=test_example['text'])
        except KeyError as e:
            print(f"Error formatting few-shot prompt for {dataset_name}: {e}")
            return f"Error formatting prompt: missing {e}"
    
    return "Unsupported dataset format"

def generate_response(model, tokenizer, prompt: str, max_length: int = 2048) -> str:
    """Generate response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated part
    generated_text = response[len(prompt):].strip()
    return generated_text

def extract_answer(response: str, dataset_name: str) -> str:
    """Extract the final answer from model response."""
    response = response.strip()
    
    if dataset_name == 'mmlu':
        # Look for pattern like "(0)" or "0" or "A"
        match = re.search(r'\(([0-3])\)|^([0-3])|\b([ABCD])\b', response)
        if match:
            if match.group(1):
                return match.group(1)
            elif match.group(2):
                return match.group(2)
            elif match.group(3):
                return str(ord(match.group(3)) - ord('A'))
        return "0"  # Default
    
    elif dataset_name == 'bb':
        # Look for Yes/No
        if 'yes' in response.lower():
            return 'Yes'
        elif 'no' in response.lower():
            return 'No'
        return 'No'  # Default
    
    elif dataset_name == 'gsm8k':
        # Look for #### pattern or final number
        match = re.search(r'####\s*([0-9,]+)', response)
        if match:
            return match.group(1).replace(',', '')
        # Look for last number in the response
        numbers = re.findall(r'\b\d+\b', response)
        if numbers:
            return numbers[-1]
        return "0"  # Default
    
    elif dataset_name == 'sst2':
        response_lower = response.lower()
        if 'positive' in response_lower:
            return 'positive'
        elif 'negative' in response_lower:
            return 'negative'
        return 'negative'  # Default
    
    elif dataset_name == 'sst5':
        response_lower = response.lower()
        if 'very positive' in response_lower:
            return 'very positive'
        elif 'very negative' in response_lower:
            return 'very negative'
        elif 'positive' in response_lower:
            return 'positive'
        elif 'negative' in response_lower:
            return 'negative'
        elif 'neutral' in response_lower:
            return 'neutral'
        return 'neutral'  # Default

def get_ground_truth(example: Dict, dataset_name: str) -> str:
    """Extract ground truth answer."""
    if dataset_name == 'mmlu':
        return str(example['answer'])
    elif dataset_name == 'bb':
        return "Yes" if example['multiple_choice_targets'][0] == "Yes" else "No"
    elif dataset_name == 'gsm8k':
        # Extract number from answer
        match = re.search(r'####\s*([0-9,]+)', example['answer'])
        if match:
            return match.group(1).replace(',', '')
        return "0"
    elif dataset_name in ['sst2', 'sst5']:
        return example['label_text']

def run_inference(selection_file_path: str, model_name: str, output_dir: str = "results"):
    """Run inference on a selection file with specified model."""
    # Load selection data
    selection_data = load_selection_file(selection_file_path)
    
    # Load model
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    model_config = MODEL_CONFIGS[model_name]
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # Run inference
    dataset_name = selection_data['dataset']
    method = selection_data['method']
    test_examples = selection_data['test_examples']
    templates = selection_data.get('templates', {})
    
    # Ensure we have templates
    if not templates:
        templates = get_default_templates(dataset_name)
        print(f"Using default templates for {dataset_name}")
    
    predictions = []
    ground_truths = []
    
    print(f"Running inference on {len(test_examples)} test examples...")
    
    # Standard inference for all methods
    for i, test_example in enumerate(test_examples):
        if (i + 1) % 10 == 0:
            print(f"Processing example {i + 1}/{len(test_examples)}")
        
        # Get selected examples for this test case
        if method == 'zero_shot':
            selected_examples = []
        elif method == 'random_shot':
            selected_examples = selection_data['examples']
        elif method in ['similarity_baseline', 'hybrid_similarity_diversity']:
            selected_examples = selection_data['examples_per_test'][i]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Format prompt
        prompt = format_prompt_for_inference(test_example, selected_examples, templates, dataset_name)
        
        # Generate response
        response = generate_response(model, tokenizer, prompt, model_config['max_length'])
        
        # Extract prediction
        predicted_answer = extract_answer(response, dataset_name)
        predictions.append(predicted_answer)
        
        # Get ground truth
        ground_truth = get_ground_truth(test_example, dataset_name)
        ground_truths.append(ground_truth)
    
    # Calculate metrics
    results = calculate_metrics(predictions, ground_truths, dataset_name)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result_file = f"{dataset_name}_{method}_{model_name}_results.json"
    result_path = os.path.join(output_dir, result_file)
    
    result_data = {
        'dataset': dataset_name,
        'method': method,
        'model': model_name,
        'predictions': predictions,
        'ground_truths': ground_truths,
        'metrics': results
    }
    
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"Results saved to: {result_path}")
    print(f"Metrics: {results}")
    
    return result_path, results

def calculate_metrics(predictions: List[str], ground_truths: List[str], dataset_name: str) -> Dict:
    """Calculate appropriate metrics based on dataset type."""
    results = {}
    
    if dataset_name in ['mmlu', 'bb', 'gsm8k', 'sst2']:
        # Binary/QA classification - use accuracy
        accuracy = accuracy_score(ground_truths, predictions)
        results['accuracy'] = accuracy
        results['metric_type'] = 'accuracy'
    
    elif dataset_name == 'sst5':
        # Multi-class classification - use F1
        f1_macro = f1_score(ground_truths, predictions, average='macro')
        f1_weighted = f1_score(ground_truths, predictions, average='weighted')
        accuracy = accuracy_score(ground_truths, predictions)
        
        results['f1_macro'] = f1_macro
        results['f1_weighted'] = f1_weighted
        results['accuracy'] = accuracy
        results['metric_type'] = 'f1'
    
    return results

def plot_results(results_dir: str = "results", output_dir: str = "plots"):
    """Plot accuracy comparison and confusion matrices."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all result files
    all_results = []
    for filename in os.listdir(results_dir):
        if filename.endswith('_results.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                data = json.load(f)
                all_results.append(data)
    
    if not all_results:
        print("No result files found!")
        return
    
    # Group by dataset
    datasets = {}
    for result in all_results:
        dataset = result['dataset']
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(result)
    
    # Plot accuracy/F1 for each dataset
    for dataset_name, dataset_results in datasets.items():
        plot_dataset_comparison(dataset_results, dataset_name, output_dir)
        
        # Plot confusion matrix for SST5
        if dataset_name == 'sst5':
            plot_confusion_matrices(dataset_results, dataset_name, output_dir)

def plot_dataset_comparison(dataset_results: List[Dict], dataset_name: str, output_dir: str):
    """Plot comparison chart for a single dataset with different colors for each method."""
    # Prepare data for plotting
    methods = []
    models = []
    scores = []
    
    # Determine metric and chart type based on dataset
    if dataset_name in ['mmlu', 'bb', 'gsm8k']:
        metric_key = 'accuracy'
        metric_name = 'Accuracy'
        chart_type = 'QA'
    elif dataset_name == 'sst2':
        metric_key = 'accuracy' 
        metric_name = 'Accuracy'
        chart_type = 'Classification'
    elif dataset_name == 'sst5':
        metric_key = 'f1_macro'
        metric_name = 'F1 Score (Macro)'
        chart_type = 'Classification'
    else:
        metric_key = 'accuracy'
        metric_name = 'Accuracy'
        chart_type = 'QA'
    
    for result in dataset_results:
        methods.append(result['method'])
        # Extract model size from model name for cleaner labels
        model_name = result['model']
        if 'qwen' in model_name.lower():
            model_size = model_name.replace('qwen', '').replace('b', 'B')
            models.append(f"Qwen {model_size}")
        else:
            models.append(model_name)
        scores.append(result['metrics'][metric_key])
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Method': methods,
        'Model': models,
        'Score': scores
    })
    
    # Define colors for each method
    method_colors = {
        'zero_shot': '#FF6B6B',      # Red
        'random_shot': '#4ECDC4',    # Teal  
        'similarity_baseline': '#45B7D1',  # Blue
        'hybrid_similarity_diversity': '#96CEB4',  # Green
        'hybrid': '#96CEB4'  # Green (alias)
    }
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Get unique methods and models
    unique_methods = df['Method'].unique()
    unique_models = df['Model'].unique()
    
    # Set up bar positions
    n_methods = len(unique_methods)
    n_models = len(unique_models)
    bar_width = 0.15
    index = np.arange(n_models)
    
    # Plot bars for each method
    for i, method in enumerate(unique_methods):
        method_data = df[df['Method'] == method]
        values = []
        for model in unique_models:
            model_data = method_data[method_data['Model'] == model]
            if len(model_data) > 0:
                values.append(model_data['Score'].iloc[0])
            else:
                values.append(0)
        
        color = method_colors.get(method, '#999999')  # Default gray if method not found
        bars = plt.bar(index + i * bar_width, values, bar_width, 
                      label=method.replace('_', ' ').title(), color=color, alpha=0.8)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Customize the plot
    plt.xlabel('Model Size', fontsize=12, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12, fontweight='bold')
    plt.title(f'{dataset_name.upper()} - {metric_name} Comparison ({chart_type})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(index + bar_width * (n_methods-1)/2, unique_models, fontsize=11)
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, min(1.1, max(scores) * 1.1))  # Set y-axis limit
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{dataset_name}_{chart_type.lower()}_{metric_key}_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {chart_type} comparison plot: {plot_path}")

def plot_confusion_matrices(dataset_results: List[Dict], dataset_name: str, output_dir: str):
    """Plot confusion matrices for SST5 dataset."""
    n_results = len(dataset_results)
    cols = min(3, n_results)
    rows = (n_results + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_results == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    
    for i, result in enumerate(dataset_results):
        predictions = result['predictions']
        ground_truths = result['ground_truths']
        
        # Create confusion matrix
        cm = confusion_matrix(ground_truths, predictions, labels=labels)
        
        # Plot
        ax = axes[i] if i < len(axes) else None
        if ax is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_title(f"{result['method']} - {result['model']}")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
    
    # Hide empty subplots
    for i in range(n_results, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{dataset_name}_confusion_matrices.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrices: {plot_path}")

def run_full_evaluation(selected_examples_dir: str = "selected_examples", 
                       models: List[str] = None, output_dir: str = "results"):
    """Run inference and evaluation for all selection files and models."""
    if models is None:
        models = ['qwen0.6b', 'qwen1.7b']
    # Find all selection files
    selection_files = []
    for filename in os.listdir(selected_examples_dir):
        if filename.endswith('.pkl'):
            selection_files.append(os.path.join(selected_examples_dir, filename))
    
    print(f"Found {len(selection_files)} selection files")
    print(f"Models to evaluate: {models}")
    
    # Run inference for each combination
    all_results = []
    for selection_file in selection_files:
        for model_name in models:
            try:
                print(f"\nRunning inference: {os.path.basename(selection_file)} with {model_name}")
                result_path, metrics = run_inference(selection_file, model_name, output_dir)
                all_results.append({
                    'selection_file': selection_file,
                    'model': model_name,
                    'result_path': result_path,
                    'metrics': metrics
                })
            except Exception as e:
                print(f"Error processing {selection_file} with {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nCompleted {len(all_results)} evaluations")
    
    # Generate plots
    if all_results:
        print("Generating plots...")
        plot_results(output_dir)
    
    return all_results

def create_results_summary(results_dir: str = "results", output_file: str = "summary_results.csv"):
    """Create a summary CSV of all results."""
    all_results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('_results.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                data = json.load(f)
                row = {
                    'dataset': data['dataset'],
                    'method': data['method'],
                    'model': data['model']
                }
                row.update(data['metrics'])
                all_results.append(row)
    
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(results_dir, output_file), index=False)
        print(f"Summary saved to: {os.path.join(results_dir, output_file)}")
        return df
    else:
        print("No results found!")
        return None

if __name__ == "__main__":
    # Example usage
    print("Running full evaluation pipeline...")
    
    # Run inference and evaluation
    results = run_full_evaluation(
        selected_examples_dir="selected_examples",
        models=['qwen0.6b', 'qwen1.7b'], 
        output_dir="results"
    )
    
    # Create summary
    summary_df = create_results_summary("results")
    
    if summary_df is not None:
        print("\nResults Summary:")
        print(summary_df.to_string(index=False))
        
        # Print best performing methods per dataset
        print("\nBest performing methods per dataset:")
        for dataset in summary_df['dataset'].unique():
            dataset_df = summary_df[summary_df['dataset'] == dataset]
            metric_col = 'f1_macro' if dataset == 'sst5' else 'accuracy'
            if metric_col in dataset_df.columns:
                best_row = dataset_df.loc[dataset_df[metric_col].idxmax()]
                print(f"{dataset}: {best_row['method']} with {best_row['model']} ({metric_col}: {best_row[metric_col]:.4f})")