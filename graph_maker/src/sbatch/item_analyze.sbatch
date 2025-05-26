#!/bin/bash
#SBATCH --job-name=master_analysis       # Название задачи
#SBATCH --error=master_analysis-%j.err   # Файл для вывода ошибок
#SBATCH --output=master_analysis-%j.log  # Файл для вывода результатов
#SBATCH --time=120:00:00                 # Максимальное время выполнения задачи (может потребоваться корректировка)
#SBATCH --cpus-per-task=1                # Для master-скрипта достаточно 1 CPU (это CPU для самой мастер-задачи Slurm, а не для дочерних)
#SBATCH --mail-user=mmnima@edu.hse.ru    # E-mail для уведомлений
#SBATCH --mail-type=ALL                  # Типы уведомлений: ALL (BEGIN, END, FAIL и др.)
#SBATCH --nodes=1                        # Использовать 1 узел
#SBATCH --array=0-30%32                  # Запускаем 31 задачу (0-30), до 32 одновременно (используя все 32 CPU)
#SBATCH --account=proj_1685


NUM_CHUNKS=31 

CONDA_ENV_PATH="/home/mmnima/.conda/envs/diplom_3"
SCRIPT_DIR="/home/mmnima/jokes/graph_maker"

echo "Master script started."
echo "Number of chunks set to: $NUM_CHUNKS"
echo "Array job concurrency: ${SLURM_ARRAY_TASK_COUNT} tasks in parallel." 

OUTPUT_ROOT_DIR="/home/mmnima/jokes/graph_maker/analysis_results_LONG_CHOPPED"
if [ ! -d "$OUTPUT_ROOT_DIR" ]; then
    mkdir -p "$OUTPUT_ROOT_DIR"
    echo "Created root output directory: $OUTPUT_ROOT_DIR"
fi


echo "Submitting individual chunk analysis jobs..."
for i in $(seq 0 $((NUM_CHUNKS-1))); do
    srun --ntasks=1 --cpus-per-task=1 -J "chunk_${i}" "$SCRIPT_DIR/run_chunk_analysis.sh" "$i" "$NUM_CHUNKS" &
done

echo "Waiting for all chunk analysis jobs to complete..."
wait

echo "All chunk analysis jobs finished. Starting results aggregation and visualization."

module load Python/Anaconda_v11.2021

source activate diplom_3

/home/mmnima/.conda/envs/diplom_3/bin/python -u "$SCRIPT_DIR/aggregate_and_visualize.py"

deactivate

echo "Results aggregation and visualization finished."
echo "Master script completed."