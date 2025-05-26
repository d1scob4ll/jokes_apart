#!/bin/bash
#SBATCH --job-name=item_analyze_array
#SBATCH --error=item_analyze_array-%A_%a.err   # %A for Job ID, %a for Array Task ID
#SBATCH --output=item_analyze_array-%A_%a.log
#SBATCH --time=05:00:00                 # Adjust time per chunk (e.g., if 10 chunks, total time was 120h, maybe 12-15h per chunk, add buffer)
#SBATCH --cpus-per-task=1              # Each task gets 1 CPU as requested
#SBATCH --nodes=1
#SBATCH --array=0-19                   

TOTAL_CHUNKS=20 

module load Python/Anaconda_v11.2021    
source activate diplom_3

echo "SLURM Job ID: $SLURM_JOB_ID, SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing chunk $SLURM_ARRAY_TASK_ID out of $TOTAL_CHUNKS total chunks."
echo "Host: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "----------------------------------"
echo "Active Conda environment:"
conda env list
echo "----------------------------------"
echo "Number of processors available to this task (nproc):"
nproc
echo "----------------------------------"

/home/mmnima/.conda/envs/diplom_3/bin/python -u item_analyze_chunk.py --task_id $SLURM_ARRAY_TASK_ID --total_chunks $TOTAL_CHUNKS

conda deactivate
echo "Chunk processing task $SLURM_ARRAY_TASK_ID finished."