"""
Example selection methods for in-context learning.
Saves selected examples to loadable files with prompt templates.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import random
from datasets import load_dataset

def get_prompt_templates():
    """Define prompt templates for different dataset types."""
    templates = {
        'mmlu': {
            'zero_shot': "Question: {question}\nChoices: {choices}\nAnswer:",
            'few_shot': "{examples}\n\nQuestion: {question}\nChoices: {choices}\nAnswer:",
            'example_format': "Question: {question}\nChoices: {choices}\nAnswer: {answer}"
        },
        'bb': {
            'zero_shot': "Statement: {statement}\nAnswer (Yes/No):",
            'few_shot': "{examples}\n\nStatement: {statement}\nAnswer (Yes/No):",
            'example_format': "Statement: {statement}\nAnswer: {answer}"
        },
        'gsm8k': {
            'zero_shot': "Problem: {problem}\nSolution:",
            'few_shot': "{examples}\n\nProblem: {problem}\nSolution:",
            'example_format': "Problem: {problem}\nSolution: {solution}"
        },
        'sst2': {
            'zero_shot': "Text: {text}\nSentiment (positive/negative):",
            'few_shot': "{examples}\n\nText: {text}\nSentiment (positive/negative):",
            'example_format': "Text: {text}\nSentiment: {sentiment}"
        },
        'sst5': {
            'zero_shot': "Text: {text}\nSentiment (very negative/negative/neutral/positive/very positive):",
            'few_shot': "{examples}\n\nText: {text}\nSentiment (very negative/negative/neutral/positive/very positive):",
            'example_format': "Text: {text}\nSentiment: {sentiment}"
        }
    }
    return templates

def load_datasets():
    """Load all datasets and format them consistently."""
    datasets = {}
    # MMLU
    df_mmlu = pd.read_parquet("hf://datasets/cais/mmlu/all/test-00000-of-00001.parquet")
    datasets['mmlu'] = {
        'data': df_mmlu,
        'input_col': 'question',
        'output_col': 'answer',
        'extra_cols': ['choices', 'subject']
    }
    
    # BigBench
    df_bb = pd.read_parquet("hf://datasets/tasksource/bigbench/anachronisms/train-00000-of-00001.parquet")
    datasets['bb'] = {
        'data': df_bb,
        'input_col': 'inputs',
        'output_col': 'multiple_choice_targets',
        'extra_cols': []
    }
    
    # GSM8K
    df_gsm8k = pd.read_parquet("hf://datasets/openai/gsm8k/main/train-00000-of-00001.parquet")
    datasets['gsm8k'] = {
        'data': df_gsm8k,
        'input_col': 'question',
        'output_col': 'answer',
        'extra_cols': []
    }
    
    # SST2
    ds_sst2 = load_dataset("SetFit/sst2")
    dfs_sst2 = []
    for split in ds_sst2.keys():
        split_df = ds_sst2[split].to_pandas()
        split_df['split'] = split
        dfs_sst2.append(split_df)
    df_sst2 = pd.concat(dfs_sst2, ignore_index=True)
    datasets['sst2'] = {
        'data': df_sst2,
        'input_col': 'text',
        'output_col': 'label_text',
        'extra_cols': []
    }
    
    # SST5
    ds_sst5 = load_dataset("SetFit/sst5")
    dfs_sst5 = []
    for split in ds_sst5.keys():
        split_df = ds_sst5[split].to_pandas()
        split_df['split'] = split
        dfs_sst5.append(split_df)
    df_sst5 = pd.concat(dfs_sst5, ignore_index=True)
    datasets['sst5'] = {
        'data': df_sst5,
        'input_col': 'text',
        'output_col': 'label_text',
        'extra_cols': []
    }
    
    return datasets

def format_example(example_data: Dict, dataset_name: str, templates: Dict) -> str:
    """Format a single example according to dataset template."""
    template = templates[dataset_name]['example_format']
    
    if dataset_name == 'mmlu':
        choices_str = ', '.join([f"({i}) {choice}" for i, choice in enumerate(example_data['choices'])])
        return template.format(
            question=example_data['question'],
            choices=choices_str,
            answer=f"({example_data['answer']}) {example_data['choices'][example_data['answer']]}"
        )
    elif dataset_name == 'bb':
        answer = "Yes" if example_data['multiple_choice_targets'][0] == "Yes" else "No"
        return template.format(
            statement=example_data['inputs'],
            answer=answer
        )
    elif dataset_name == 'gsm8k':
        return template.format(
            problem=example_data['question'],
            solution=example_data['answer']
        )
    elif dataset_name in ['sst2', 'sst5']:
        # FIXED: Map 'label_text' from data to 'sentiment' in template
        return template.format(
            text=example_data['text'],
            sentiment=example_data['label_text']
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def zero_shot_selection(test_examples: List[Dict], dataset_name: str, output_dir: str):
    """Zero-shot baseline - no examples selected."""
    templates = get_prompt_templates()
    
    selection_data = {
        'method': 'zero_shot',
        'dataset': dataset_name,
        'examples': [],
        'templates': templates[dataset_name],
        'test_examples': test_examples
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_zero_shot.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(selection_data, f)
    
    print(f"Saved zero-shot selection to {output_path}")
    return output_path

def random_shot_selection(demo_pool: List[Dict], test_examples: List[Dict], 
                         dataset_name: str, k: int, output_dir: str):
    """Random example selection baseline."""
    templates = get_prompt_templates()
    
    selected_examples = random.sample(demo_pool, min(k, len(demo_pool)))
    
    selection_data = {
        'method': 'random_shot',
        'dataset': dataset_name,
        'k': k,
        'examples': selected_examples,
        'templates': templates[dataset_name],
        'test_examples': test_examples
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_random_shot_k{k}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(selection_data, f)
    
    print(f"Saved random-shot selection to {output_path}")
    return output_path

def similarity_baseline_selection(demo_pool: List[Dict], test_examples: List[Dict],
                                dataset_name: str, k: int, output_dir: str,
                                encoder_name: str = "all-MiniLM-L6-v2"):
    """Similarity-based selection using sentence transformers + KNN/Cosine similarity."""
    templates = get_prompt_templates()
    encoder = SentenceTransformer(encoder_name)
    
    # Extract input texts for encoding
    demo_texts = []
    test_texts = []
    
    for demo in demo_pool:
        if dataset_name == 'mmlu':
            demo_texts.append(demo['question'])
        elif dataset_name == 'bb':
            demo_texts.append(demo['inputs'])
        elif dataset_name == 'gsm8k':
            demo_texts.append(demo['question'])
        elif dataset_name in ['sst2', 'sst5']:
            demo_texts.append(demo['text'])
    
    for test in test_examples:
        if dataset_name == 'mmlu':
            test_texts.append(test['question'])
        elif dataset_name == 'bb':
            test_texts.append(test['inputs'])
        elif dataset_name == 'gsm8k':
            test_texts.append(test['question'])
        elif dataset_name in ['sst2', 'sst5']:
            test_texts.append(test['text'])
    
    # Encode all texts
    print("Encoding demonstration pool...")
    demo_embeddings = encoder.encode(demo_texts)
    print("Encoding test examples...")
    test_embeddings = encoder.encode(test_texts)
    
    # For each test example, find top-k most similar demonstrations
    selections_per_test = []
    for i, test_emb in enumerate(test_embeddings):
        similarities = cosine_similarity([test_emb], demo_embeddings)[0]
        top_k_indices = np.argsort(similarities)[::-1][:k]
        selected_demos = [demo_pool[idx] for idx in top_k_indices]
        selections_per_test.append(selected_demos)
    
    selection_data = {
        'method': 'similarity_baseline',
        'dataset': dataset_name,
        'k': k,
        'encoder_name': encoder_name,
        'examples_per_test': selections_per_test,
        'templates': templates[dataset_name],
        'test_examples': test_examples
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_similarity_baseline_k{k}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(selection_data, f)
    
    print(f"Saved similarity baseline selection to {output_path}")
    return output_path



def hybrid_similarity_diversity_selection(demo_pool: List[Dict], test_examples: List[Dict],
                                        dataset_name: str, k: int, output_dir: str,
                                        encoder_name: str = "all-MiniLM-L6-v2"):
    """Hybrid similarity-diversity selection using clustering."""
    templates = get_prompt_templates()
    encoder = SentenceTransformer(encoder_name)
    
    # Extract demo texts
    demo_texts = []
    for demo in demo_pool:
        if dataset_name == 'mmlu':
            demo_texts.append(demo['question'])
        elif dataset_name == 'bb':
            demo_texts.append(demo['inputs'])
        elif dataset_name == 'gsm8k':
            demo_texts.append(demo['question'])
        elif dataset_name in ['sst2', 'sst5']:
            demo_texts.append(demo['text'])
    
    print("Encoding demonstration pool...")
    demo_embeddings = encoder.encode(demo_texts)
    
    # Cluster demonstrations into k clusters
    print(f"Clustering into {k} clusters...")
    kmeans = KMeans(n_clusters=min(k, len(demo_pool)), random_state=42)
    cluster_labels = kmeans.fit_predict(demo_embeddings)
    
    # Select representative from each cluster
    representatives = []
    for cluster_id in range(kmeans.n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        
        # Find closest demo to cluster center
        distances = np.linalg.norm(demo_embeddings[cluster_indices] - cluster_center, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        representatives.append(closest_idx)
    
    # For each test example, rank representatives by similarity
    selections_per_test = []
    test_texts = []
    for test in test_examples:
        if dataset_name == 'mmlu':
            test_texts.append(test['question'])
        elif dataset_name == 'bb':
            test_texts.append(test['inputs'])
        elif dataset_name == 'gsm8k':
            test_texts.append(test['question'])
        elif dataset_name in ['sst2', 'sst5']:
            test_texts.append(test['text'])
    
    print("Encoding test examples...")
    test_embeddings = encoder.encode(test_texts)
    
    for test_emb in test_embeddings:
        rep_embeddings = demo_embeddings[representatives]
        similarities = cosine_similarity([test_emb], rep_embeddings)[0]
        ranked_indices = np.argsort(similarities)[::-1]
        
        selected_demos = [demo_pool[representatives[idx]] for idx in ranked_indices]
        selections_per_test.append(selected_demos)
    
    selection_data = {
        'method': 'hybrid_similarity_diversity',
        'dataset': dataset_name,
        'k': k,
        'encoder_name': encoder_name,
        'examples_per_test': selections_per_test,
        'templates': templates[dataset_name],
        'test_examples': test_examples
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_hybrid_k{k}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(selection_data, f)
    
    print(f"Saved hybrid selection to {output_path}")
    return output_path

def get_optimal_split_ratio(total_samples: int, min_demo: int = 100, min_test: int = 50):
    """Calculate optimal demo/test split ratio based on total dataset size."""
    if total_samples < min_demo + min_test:
        # If dataset too small, use 70/30 split
        demo_size = int(total_samples * 0.7)
        test_size = total_samples - demo_size
        return demo_size, test_size
    elif total_samples < 1000:
        # Small datasets: use 80/20 split
        demo_size = int(total_samples * 0.8)
        test_size = total_samples - demo_size
        return demo_size, test_size
    elif total_samples < 10000:
        # Medium datasets: use 90/10 split but cap test at 1000
        test_size = min(int(total_samples * 0.1), 1000)
        demo_size = total_samples - test_size
        return demo_size, test_size
    else:
        # Large datasets: use fixed test size
        test_size = 1000
        demo_size = total_samples - test_size
        return demo_size, test_size
    
def run_example_selection(dataset_name: str, k: int = 16, output_dir: str = "selected_examples"):
    """Main function to run all example selection methods for a dataset."""
    print(f"Running example selection for {dataset_name}...")
    # Load dataset
    datasets = load_datasets()
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_info = datasets[dataset_name]
    df = dataset_info['data']
    
    # Convert to list of dicts
    all_examples = df.to_dict('records')
    random.shuffle(all_examples)
    
    # Calculate optimal split
    demo_size, test_size = get_optimal_split_ratio(len(all_examples))
    print(f"Dataset size: {len(all_examples)}")
    print(f"Using demo_size: {demo_size}, test_size: {test_size}")
    
    # Split into demo pool and test set
    demo_pool = all_examples[:demo_size]
    test_examples = all_examples[demo_size:demo_size + test_size]
    
    print(f"Demo pool size: {len(demo_pool)}")
    print(f"Test set size: {len(test_examples)}")
    
    # Run all selection methods
    methods = {}
    
    # Zero-shot
    methods['zero_shot'] = zero_shot_selection(test_examples, dataset_name, output_dir)
    
    # Random-shot
    methods['random_shot'] = random_shot_selection(demo_pool, test_examples, dataset_name, k, output_dir)
    
    # Similarity baseline
    methods['similarity_baseline'] = similarity_baseline_selection(
        demo_pool, test_examples, dataset_name, k, output_dir)
    
    # Hybrid similarity-diversity
    methods['hybrid'] = hybrid_similarity_diversity_selection(
        demo_pool, test_examples, dataset_name, k, output_dir)
    
    print(f"Completed example selection for {dataset_name}")
    print("Generated files:")
    for method, path in methods.items():
        print(f"  {method}: {path}")
    
    return methods

if __name__ == "__main__":
    # Run for all datasets
    datasets = ['mmlu', 'bb', 'gsm8k', 'sst2', 'sst5']
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Processing {dataset.upper()}")
        print('='*50)
        run_example_selection(dataset)