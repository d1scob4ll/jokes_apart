import pickle
import os
import logging

# --- Настройки ---
OUTPUT_DIR_GRAPH = 'graph_data'
OUTPUT_DIR_SHORT_RELATIONS = os.path.join(OUTPUT_DIR_GRAPH, 'short_relations')
FINAL_RELATIONS_FILE = os.path.join(OUTPUT_DIR_SHORT_RELATIONS, 'short_corpus_relations.pkl')
NUM_TASKS = 14 # Убедитесь, что это число соответствует количеству задач в массиве SLURM

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting script to combine partial short corpus relations.")

all_combined_relations = []
successful_tasks = 0

# Собираем результаты из всех частичных файлов
for task_id in range(NUM_TASKS):
    partial_file = os.path.join(OUTPUT_DIR_SHORT_RELATIONS, f'short_corpus_relations_part_{task_id}.pkl')
    if os.path.exists(partial_file):
        logging.info(f"Loading partial results from {partial_file}...")
        try:
            with open(partial_file, 'rb') as f:
                partial_relations = pickle.load(f)
                all_combined_relations.extend(partial_relations)
                logging.info(f"Loaded {len(partial_relations)} relations from task {task_id}.")
                successful_tasks += 1
        except Exception as e:
            logging.error(f"Error loading partial results from {partial_file}: {e}", exc_info=True)
    else:
        logging.warning(f"Partial result file not found for task {task_id}: {partial_file}. This task might have failed or produced no output.")

logging.info(f"Finished loading partial results. Successfully loaded from {successful_tasks}/{NUM_TASKS} tasks.")
logging.info(f"Total collected relations: {len(all_combined_relations)}")

# Сохраняем объединенные результаты
if not all_combined_relations:
    logging.warning("No relations were collected from any task. The final output file will not be created.")
else:
    logging.info(f"Saving combined relations to {FINAL_RELATIONS_FILE}...")
    try:
        with open(FINAL_RELATIONS_FILE, 'wb') as f:
            pickle.dump(all_combined_relations, f)
        logging.info("Combined short corpus relations successfully saved.")

        # Опционально: удалить частичные файлы после успешного объединения
        logging.info("Cleaning up partial files...")
        for task_id in range(NUM_TASKS):
             partial_file = os.path.join(OUTPUT_DIR_SHORT_RELATIONS, f'short_corpus_relations_part_{task_id}.pkl')
             if os.path.exists(partial_file):
                 try:
                     os.remove(partial_file)
                     # logging.info(f"Removed {partial_file}") # Может быть слишком много логов
                 except Exception as e:
                     logging.warning(f"Could not remove partial file {partial_file}: {e}")
        logging.info("Partial file cleanup finished.")

    except Exception as e:
        logging.error(f"Error saving combined relations to {FINAL_RELATIONS_FILE}: {e}", exc_info=True)

logging.info("Combine results script finished.")