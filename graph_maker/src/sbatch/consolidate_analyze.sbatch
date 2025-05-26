#!/bin/bash
#SBATCH --job-name=consolidate_analysis
#SBATCH --error=consolidate_analysis-%j.err
#SBATCH --output=consolidate_analysis-%j.log
#SBATCH --time=24:00:00                
#SBATCH --cpus-per-task=4              
#SBATCH --nodes=1

TOTAL_CHUNKS=20

module load Python/Anaconda_v11.2021    
source activate diplom_3

echo "Starting consolidation and final analysis for $TOTAL_CHUNKS chunks."
echo "Host: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "----------------------------------"
echo "Active Conda environment:"
conda env list
echo "----------------------------------"

/home/mmnima/.conda/envs/diplom_3/bin/python -u consolidate_analyze.py --total_chunks $TOTAL_CHUNKS

conda deactivate
echo "Consolidation and final analysis finished."