#!/bin/bash
#SBATCH --job-name=combine_short_relations
#SBATCH --error=combine_short_relations-%j.err
#SBATCH --output=combine_short_relations-%j.log
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=mmnima@edu.hse.ru
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --dependency=afterok:2813270 # Обязательно замените 2813200 на реальный Job ID вашего массива задач!

module load Python/Anaconda_v11.2021

source activate diplom_2

echo "Проверка активного окружения Conda:"
conda env list
echo "----------------------------------"

echo "Информация о задаче SLURM:"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "----------------------------------"

python3 -u combine_short_corpus_relations.py

deactivate

echo "Combine results script finished."