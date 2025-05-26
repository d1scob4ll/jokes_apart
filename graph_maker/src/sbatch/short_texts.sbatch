#!/bin/bash
#SBATCH --job-name=short_texts_array
#SBATCH --error=short_texts_array-%A_%a.err
#SBATCH --output=short_texts_array-%A_%a.log
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=mmnima@edu.hse.ru
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --array=0-13

TOTAL_LINES=344449
NUM_TASKS=14
TASK_ID=$SLURM_ARRAY_TASK_ID

module load Python/Anaconda_v11.2021

source activate diplom_2

echo "Проверка активного окружения Conda:"
conda env list
echo "----------------------------------"

echo "Информация о задаче SLURM:"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Total Lines: $TOTAL_LINES"
echo "Number of Tasks: $NUM_TASKS"
echo "----------------------------------"

nproc
echo "----------------------------------"

ls
echo "----------------------------------"

python3 -u process_short_corpus.py \
  --total_lines $TOTAL_LINES \
  --task_id $TASK_ID \
  --num_tasks $NUM_TASKS

deactivate

echo "SLURM Task $TASK_ID finished."