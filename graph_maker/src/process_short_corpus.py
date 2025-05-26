import spacy
import logging
import os
import re
import pickle
import argparse # Импортируем argparse для обработки аргументов

from utils import get_syntactic_relations, contains_only_russian_or_latin_letters, sanitize_string_for_xml

SPACY_MODEL = "ru_core_news_sm"

ANEK_FILE = '/home/mmnima/jokes/aneks.txt'
NOT_ANEKS_FILE = '/home/mmnima/jokes/not_aneks.txt' # Этот файл пока оставим закомментированным в логике использования
OUTPUT_DIR_GRAPH = 'graph_data'
OUTPUT_DIR_SHORT_RELATIONS = os.path.join(OUTPUT_DIR_GRAPH, 'short_relations')
# Имя файла для сохранения отношений будет зависеть от ID задачи
# SHORT_RELATIONS_FILE = os.path.join(OUTPUT_DIR_SHORT_RELATIONS, 'short_corpus_relations.pkl')

SHORT_BATCH_SIZE = 50
PIPE_N_PROCESS = 1 # Для распараллеливания по данным, n_process в nlp.pipe лучше оставить 1

# --- Обработка аргументов командной строки ---
parser = argparse.ArgumentParser(description='Process a chunk of the short corpus for graph building.')
parser.add_argument('--total_lines', type=int, required=True, help='Total number of lines in the input file.')
parser.add_argument('--task_id', type=int, required=True, help='ID of the current task (0-indexed).')
parser.add_argument('--num_tasks', type=int, required=True, help='Total number of tasks.')
args = parser.parse_args()

total_lines = args.total_lines
task_id = args.task_id
num_tasks = args.num_tasks

# Рассчитываем диапазон строк для текущей задачи
lines_per_task = total_lines // num_tasks
remainder = total_lines % num_tasks

# Определяем начальный и конечный индекс строки для этой задачи (0-indexed)
start_index = task_id * lines_per_task + min(task_id, remainder)
end_index = (task_id + 1) * lines_per_task + min(task_id + 1, remainder) - 1

# Устанавливаем имя файла для сохранения частичных результатов
PARTIAL_RELATIONS_FILE = os.path.join(OUTPUT_DIR_SHORT_RELATIONS, f'short_corpus_relations_part_{task_id}.pkl')

# --- Создание директорий ---
if not os.path.exists(OUTPUT_DIR_GRAPH):
    os.makedirs(OUTPUT_DIR_GRAPH)
if not os.path.exists(OUTPUT_DIR_SHORT_RELATIONS):
    os.makedirs(OUTPUT_DIR_SHORT_RELATIONS)

# --- Настройка логирования ---
# Добавим ID задачи в формат логирования для отслеживания
logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - Task {task_id} - %(levelname)s - %(message)s')

# --- Загрузка модели SpaCy ---
try:
    nlp = spacy.load(SPACY_MODEL, disable=["textcat"])
    logging.info(f"Модель SpaCy '{SPACY_MODEL}' успешно загружена.")
except OSError:
    logging.error(f"Модель SpaCy '{SPACY_MODEL}' не найдена. Пожалуйста, скачайте ее: python -m spacy download {SPACY_MODEL}. Exiting.", exc_info=True)
    exit()

# --- Чтение данных из aneks.txt ---
anek_lines_chunk = []
if os.path.exists(ANEK_FILE):
    logging.info(f"Reading lines from {ANEK_FILE} for task {task_id}. Processing lines from {start_index} to {end_index}...")
    try:
        with open(ANEK_FILE, 'r', encoding='utf-8') as f:
            # Читаем только нужный диапазон строк
            for i, line in enumerate(f):
                if i > end_index:
                    break # Останавливаем чтение после нужного диапазона
                if i >= start_index:
                    stripped_line = line.strip()
                    if stripped_line:
                         anek_lines_chunk.append(stripped_line)

        logging.info(f"Task {task_id} read {len(anek_lines_chunk)} lines from {ANEK_FILE} (indices {start_index}-{end_index}).")
    except Exception as e:
        logging.error(f"Error reading file {ANEK_FILE} for task {task_id}: {e}", exc_info=True)
else:
    logging.error(f"Anekdot file not found: {ANEK_FILE}. Cannot proceed. Exiting task {task_id}.")
    exit() # Задача должна завершиться, если файла нет

# --- Игнорируем not_aneks.txt для этой параллельной обработки ---
# sentence_corpus_lines_graph = []
# if os.path.exists(NOT_ANEKS_FILE):
# ... (код чтения not_aneks.txt закомментирован) ...
# else:
#     logging.warning(f"Not-anekdot file not found: {NOT_ANEKS_FILE}. Cannot use sentence corpus for graph building.")

# --- Подготовка данных для SpaCy pipe ---
short_texts_for_pipe = []
short_text_ids = []

# Используем только прочитанный чанк из aneks.txt
if anek_lines_chunk:
    logging.info(f"Task {task_id}: Preparing {len(anek_lines_chunk)} lines from anek file for SpaCy processing.")
    for i, line in enumerate(anek_lines_chunk):
        # Сквозной индекс строки в исходном файле = start_index + i
        original_index = start_index + i
        if line:
            short_texts_for_pipe.append(line)
            # Используем оригинальный индекс строки для ID
            short_text_ids.append(f"anek_graph_{original_index}")

if not short_texts_for_pipe:
    logging.error(f"Task {task_id}: Short texts list is empty for this chunk ({start_index}-{end_index}). Nothing to process. Exiting.")
    exit() # Задача завершается, если нет данных для обработки

logging.info(f"\nTask {task_id}: Starting SpaCy parallel processing of {len(short_texts_for_pipe)} short texts using nlp.pipe (n_process={PIPE_N_PROCESS}, batch_size={SHORT_BATCH_SIZE}).")

all_relations_list_short = []

# Обрабатываем только тексты из текущего чанка
processed_short_docs_iterator = nlp.pipe(short_texts_for_pipe, n_process=PIPE_N_PROCESS, batch_size=SHORT_BATCH_SIZE)

for i, doc in enumerate(processed_short_docs_iterator):
    try:
        # current_text_id = short_text_ids[i] # ID уже содержит оригинальный индекс

        chunk_relations = get_syntactic_relations(doc)

        if chunk_relations:
            # Добавляем ID оригинального текста к каждой связи, если нужно отслеживать источник
            # (Зависит от структуры get_syntactic_relations и желаемого формата выходных данных)
            # Если get_syntactic_relations возвращает список кортежей/объектов связей,
            # возможно, вам захочется добавить ID:
            # relations_with_id = [(short_text_ids[i], rel) for rel in chunk_relations]
            # all_relations_list_short.extend(relations_with_id)
            # Если relations_list_short - это просто список связей без привязки к ID исходного текста,
            # тогда просто добавляем:
             all_relations_list_short.extend(chunk_relations)


        # Логирование прогресса по чанкам этой задачи
        if (i + 1) % 1000 == 0 or (i + 1) == len(short_texts_for_pipe):
            logging.info(f"Task {task_id}: Completed SpaCy processing for {i+1}/{len(short_texts_for_pipe)} texts in chunk.")
    except Exception as e:
        # Логирование ошибки с указанием индекса в *текущем* чанке и оригинального ID
        error_original_id = short_text_ids[i] if i < len(short_text_ids) else f"index_in_chunk_{i}_unknown_original_id"
        logging.error(f"Task {task_id}: Error processing short text index in chunk {i} (Original ID: {error_original_id}) with SpaCy: {e}", exc_info=True)

logging.info(f"Task {task_id}: Finished SpaCy parallel processing for its chunk.")

# --- Сохранение частичных результатов ---
if not all_relations_list_short:
    logging.warning(f"Task {task_id}: No relations extracted from its chunk. Nothing to save for this task.")
else:
    logging.info(f"Task {task_id}: Extracted {len(all_relations_list_short)} relations from its chunk. Saving to {PARTIAL_RELATIONS_FILE}...")
    try:
        with open(PARTIAL_RELATIONS_FILE, 'wb') as f:
            pickle.dump(all_relations_list_short, f)
        logging.info(f"Task {task_id}: Partial relations successfully saved to {PARTIAL_RELATIONS_FILE}.")
    except Exception as e:
        logging.error(f"Task {task_id}: Error saving partial relations to {PARTIAL_RELATIONS_FILE}: {e}", exc_info=True)

logging.info(f"Task {task_id}: Short corpus processing script finished.")