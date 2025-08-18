#!/bin/bash
#SBATCH --job-name=example_selection_icl
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=msabdussyakur1@sheffield.ac.uk

# Create logs directory
mkdir -p logs

# Load required modules
module load GCC/11.3.0 OpenMPI/4.1.4
module load Python/3.10.4
module load CUDA/11.7.0

# Print GPU information
echo "=== GPU Information ==="
nvidia-smi
echo "======================="

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION" 
echo "QOS: $SLURM_JOB_QOS"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Working Directory: $(pwd)"
echo "Temporary Directory: $TMPDIR"

# Set up Python environment
echo "Setting up Python environment..."
python -m venv venv
source venv/bin/activate

# Upgrade pip and install requirements
pip install --extra-index-url https://pypi.fury.io/arrow-nightlies/ \
        --prefer-binary --pre pyarrow
python -m pip install pandas numpy scikit-learn matplotlib seaborn sentence-transformers transformers torch datasets accelerate

# Set HuggingFace cache to use scratch space
export TRANSFORMERS_CACHE=$TMPDIR/transformers_cache
export HF_HOME=$TMPDIR/hf_home
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME

# Print Python and package versions
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run the experiment
echo "Starting example selection..."
python main.py --mode selection --datasets mmlu bb gsm8k sst2 sst5 --k 8

echo "Starting inference and evaluation..."
python main.py --mode inference --models qwen0.6b qwen1.7b qwen4b qwen8b

echo "Job completed successfully!"

# Copy results to home directory (optional, since scratch may be cleaned)
if [ -d "results" ]; then
    cp -r results $HOME/icl_experiment_results_$SLURM_JOB_ID
    echo "Results copied to: $HOME/icl_experiment_results_$SLURM_JOB_ID"
fi

if [ -d "plots" ]; then
    cp -r plots $HOME/icl_experiment_plots_$SLURM_JOB_ID
    echo "Plots copied to: $HOME/icl_experiment_plots_$SLURM_JOB_ID"
fi