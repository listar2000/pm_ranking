#!/bin/bash
#SBATCH --job-name=pm_ranking_test
#SBATCH --output=/net/scratch2/listar2000/pm_ranking/slurm/pm_ranking_test.out
#SBATCH --error=/net/scratch2/listar2000/pm_ranking/slurm/pm_ranking_test.err
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH --partition=general

nvidia-smi

cd /net/scratch2/listar2000/pm_ranking

source .venv/bin/activate

python test/test_all_models.py

