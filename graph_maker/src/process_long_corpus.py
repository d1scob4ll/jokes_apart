import spacy
import logging
import os
import re
import pickle
from joblib import Parallel, delayed

from utils import get_syntactic_relations, contains_only_russian_or_latin_letters, sanitize_string_for_xml

SPACY_MODEL = "ru_core_news_sm"

TOLSTOY_DIR = '/home/mmnima/jokes/flibusta'
OUTPUT_DIR_GRAPH = 'graph_data_3'
OUTPUT_DIR_LONG_RELATIONS = os.path.join(OUTPUT_DIR_GRAPH, 'long_relations')
LONG_RELATIONS_FILE = os.path.join(OUTPUT_DIR_LONG_RELATIONS, 'long_corpus_relations.pkl')

CHUNK_SIZE = 750000
GENERAL_CORPUS_FRACTION = 1

LONG_BATCH_SIZE = 20
PIPE_N_PROCESS = -1

if not os.path.exists(OUTPUT_DIR_GRAPH):
    os.makedirs(OUTPUT_DIR_GRAPH)
if not os.path.exists(OUTPUT_DIR_LONG_RELATIONS):
    os.makedirs(OUTPUT_DIR_LONG_RELATIONS)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    nlp = spacy.load(SPACY_MODEL, disable=["textcat"])
    logging.info(f"Модель SpaCy '{SPACY_MODEL}' успешно загружена.")
except OSError:
    logging.error(f"Модель SpaCy '{SPACY_MODEL}' не найдена. Пожалуйста, скачайте ее: python -m spacy download {SPACY_MODEL}. Exiting.", exc_info=True)
    exit()

nlp.max_length = 20000000

all_tolstoy_files_content = []
if os.path.exists(TOLSTOY_DIR):
    logging.info(f"Reading text files from {TOLSTOY_DIR}...")
    tolstoy_filenames = [f for f in os.listdir(TOLSTOY_DIR) if f.endswith(".txt") and contains_only_russian_or_latin_letters(f)]

    total_tolstoy_files = len(tolstoy_filenames)
    files_to_process_count = int(total_tolstoy_files * GENERAL_CORPUS_FRACTION)
    files_to_process_count = max(1, files_to_process_count) if total_tolstoy_files > 0 else 0
    tolstoy_filenames = tolstoy_filenames[:files_to_process_count]

    for filename in tolstoy_filenames:
        filepath = os.path.join(TOLSTOY_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                all_tolstoy_files_content.append((filename, f.read()))
            logging.info(f"Read file: {filepath}")
        except Exception as e:
            logging.error(f"Error reading file {filepath}: {e}", exc_info=True)
else:
    logging.error(f"Directory not found: {TOLSTOY_DIR}. Cannot process long corpus. Exiting.", exc_info=True)
    exit()

def process_paragraph(filename, para_index, paragraph_text, chunk_size):
    chunks = []
    chunk_ids = []
    cleaned_para = paragraph_text.strip()

    if not cleaned_para:
        return chunks, chunk_ids

    if len(cleaned_para) > chunk_size:
        chunk_counter = 0
        start_index = 0
        while start_index < len(cleaned_para):
            target_end_index = start_index + chunk_size

            if target_end_index >= len(cleaned_para):
                end_index = len(cleaned_para)
            else:
                split_point = -1
                for j in range(target_end_index, start_index - 1, -1):
                    if cleaned_para[j].isspace() or cleaned_para[j] in '.,;!?:':
                        split_point = j
                        break

                if split_point != -1:
                    end_index = split_point + 1
                else:
                    end_index = target_end_index

            chunk = cleaned_para[start_index:end_index].strip()

            if chunk:
                chunks.append(chunk)
                chunk_ids.append(f"tolstoy_{filename}_para_{para_index}_chunk_{chunk_counter}")
                chunk_counter += 1

            start_index = end_index

    else:
        if len(cleaned_para) > 0:
            chunks.append(cleaned_para)
            chunk_ids.append(f"tolstoy_{filename}_para_{para_index}")

    return chunks, chunk_ids


long_chunks_for_pipe = []
long_chunk_ids = []

if all_tolstoy_files_content:
    logging.info("Processing Tolstoy texts into chunks using joblib...")

    tasks = []
    for filename, text_content in all_tolstoy_files_content:
        paragraphs = re.split(r'\n\s*\n', text_content)
        logging.info(f"Processing file: {filename} with {len(paragraphs)} paragraphs")
        for i, para in enumerate(paragraphs):
            tasks.append(delayed(process_paragraph)(filename, i, para, CHUNK_SIZE))

    results = Parallel(n_jobs=-1)(tasks)

    total_chunks_created = 0
    for chunks, chunk_ids in results:
        long_chunks_for_pipe.extend(chunks)
        long_chunk_ids.extend(chunk_ids)
        total_chunks_created += len(chunks)

    logging.info(f"Finished processing Tolstoy. Total chunks created: {total_chunks_created}")

else:
    logging.warning("No Tolstoy texts available. Long corpus processing skipped.")
    exit()

if not long_chunks_for_pipe:
    logging.error("Long chunks list is empty after processing Tolstoy texts. Cannot proceed with SpaCy processing. Exiting.")
    exit()

logging.info(f"\nStarting SpaCy parallel processing of {len(long_chunks_for_pipe)} long chunks using nlp.pipe (n_process={PIPE_N_PROCESS}, batch_size={LONG_BATCH_SIZE}).")

all_relations_list_long = []

processed_long_docs_iterator = nlp.pipe(long_chunks_for_pipe, n_process=PIPE_N_PROCESS, batch_size=LONG_BATCH_SIZE)

for i, doc in enumerate(processed_long_docs_iterator):
    try:
        current_chunk_id = long_chunk_ids[i]

        chunk_relations = get_syntactic_relations(doc)

        if chunk_relations:
             all_relations_list_long.extend(chunk_relations)

        if (i + 1) % 1000 == 0 or (i + 1) == len(long_chunks_for_pipe):
             logging.info(f"Completed SpaCy processing for {i+1}/{len(long_chunks_for_pipe)} long chunks.")

    except Exception as e:
         error_id = long_chunk_ids[i] if i < len(long_chunk_ids) else f"index_{i}_unknown_chunk_id"
         logging.error(f"Error processing long chunk index {i} (ID: {error_id}) with SpaCy: {e}", exc_info=True)


logging.info("Finished SpaCy parallel processing for long chunks.")

if not all_relations_list_long:
     logging.warning("No relations extracted from long chunks. Nothing to save.")
else:
    logging.info(f"Extracted {len(all_relations_list_long)} relations from long chunks. Saving to {LONG_RELATIONS_FILE}...")
    try:
        with open(LONG_RELATIONS_FILE, 'wb') as f:
            pickle.dump(all_relations_list_long, f)
        logging.info("Long corpus relations successfully saved.")
    except Exception as e:
        logging.error(f"Error saving long corpus relations to {LONG_RELATIONS_FILE}: {e}", exc_info=True)

logging.info("Long corpus processing script finished.")
