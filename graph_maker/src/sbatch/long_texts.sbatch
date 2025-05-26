#!/bin/bash
#SBATCH --job-name=long_texts_2       # Название задачи
#SBATCH --error=long_texts_2-%j.err   # Файл для вывода ошибок
#SBATCH --output=long_texts_2-%j.log  # Файл для вывода результатов
#SBATCH --time=120:00:00                 # Максимальное время выполнения задачи
#SBATCH --cpus-per-task=8              # Использование 4 CPU
#SBATCH --mail-user=mmnima@edu.hse.ru   # E-mail для уведомлений
#SBATCH --mail-type=ALL                 # Типы уведомлений: ALL (BEGIN, END, FAIL и др.)
#SBATCH --nodes=1                       # Использовать 1 узел

# Загрузка необходимых модулей
module load Python/Anaconda_v11.2021    
module load julia/v1.9.0

source activate diplom_2

echo "Проверка активного окружения Conda:"
conda env list
echo "----------------------------------" # Добавим разделитель для читаемости в логе

nproc
echo "----------------------------------" 

ls
echo "----------------------------------" 

# Запуск Python-скрипта
python3 -u process_long_corpus.py                 # -u для немедленного вывода результатов

deactivate
