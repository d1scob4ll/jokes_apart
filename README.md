# jokes_apart

## Установка

Для запуска этого проекта необходимо использовать **Conda** для управления окружением. Если у вас уже установлена Anaconda или Miniconda, перейдите сразу к шагу "Клонируйте репозиторий проекта".

### 1. Установка Anaconda

Если у вас ещё нет Conda, вы можете установить полную версию Anaconda, которая включает множество полезных пакетов для работы с данными.

1.  **Скачайте установщик Anaconda:**
    ```
    wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh
    ```

2.  **Запустите установщик:**
    Следуйте инструкциям на экране. Рекомендуется использовать установки по умолчанию.
    ```
    bash Anaconda3-latest-Linux-x86_64.sh
    ```
    В процессе установки вас могут попросить нажать Enter для просмотра лицензии, а затем ввести `yes` для её принятия. Также вас спросят, хотите ли вы инициализировать Anaconda3, ответьте `yes`.

3.  **Обновите переменные окружения:**
    Эта команда перезагрузит ваш профиль оболочки, чтобы Conda стала доступна в терминале.
    ```
    source ~/.bashrc
    ```
    Если вы используете другую оболочку (например, Zsh), вам может понадобиться `source ~/.zshrc` или аналогичная команда.

### 2. Клонируйте репозиторий проекта

Перейдите в директорию, где вы хотите разместить проект, и клонируйте его:

```
git clone https://github.com/d1scob4ll/jokes_apart.git

cd jokes_apart
```

### 3. Создайте и активируйте Conda-окружение

Перейдите в корневую директорию проекта (где находится файл environment.yml) и выполните следующие команды, чтобы создать окружение на основе файла environment.yml:

```
conda env create -f environment.yml
conda activate diplom_3 # Или имя окружения, указанное в environment.yml
```

## Описание папок

### Датасет

Данные необходимо скачать из Яндекс диска и поместить в папку graph_maker: https://disk.yandex.com/d/LFf6rvvZpngESQ

Для разархивирования необходимо написать команду:

```
tar -xzvf data.tar.gz
```

Все данные находятся в `data`:

1. В `big_sample/` находится большая выборка из Флибусты с 1400 текстами.

2. В `small_sample/` находится малая выборка из Флибусты с 600 текстами.

3. В `jokes_and_not_jokes/` находится юмористический корпус в `aneks.txt`, а так же извлеченные из малой выборки предложения в `not_aneks.txt`.

4. В `generated/` находятся сгенерированные LLM предложения и шутки.

### Исходный код

Папка `src/` содержит исходный код и скрипты для обработки данных, анализа и построения графов:

1. `build_final_graph.py` — скрипт для построения итогового графа на основе обработанных данных.

2. `combine_short_corpus_relations.py` — скрипт для объединения отношений, извлеченных из короткого корпуса.

3. `consolidate_analyze.py` — скрипт для консолидации и анализа данных, включая объединение статистики и генерацию итоговых результатов.

4. `item_analyze_chunk.py` — скрипт для анализа данных по частям (чанкам), используется для параллельной обработки.

5. `process_long_corpus.py` — скрипт для обработки длинного корпуса текстов (например, извлечение отношений и статистики).

6. `process_short_corpus.py` — скрипт для обработки короткого корпуса текстов (аналогично для меньшего объема данных).

7. Папка `sbatch/` содержит скрипты для управления задачами на кластере с использованием SLURM:
   - `combine_results.sbatch` — для объединения результатов анализа.
   - `consolidate_analyze.sbatch` — для запуска консолидации и анализа.
   - `graph_cooking.sbatch` — для построения графов.
   - `item_analyze.sbatch` — для анализа отдельных элементов.
   - `item_analyze_array.sbatch` — для параллельного анализа элементов в виде массива задач.
   - `long_texts.sbatch` — для обработки длинных текстов.
   - `run_chunk_analysis.sh` — вспомогательный скрипт для запуска анализа по чанкам.
   - `short_texts.sbatch` — для обработки коротких текстов.

9. `utils.py` — модуль с вспомогательными функциями, используемыми в других скриптах.
