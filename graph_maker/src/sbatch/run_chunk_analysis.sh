#!/bin/bash
#

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <CHUNK_ID> <NUM_CHUNKS>"
  exit 1
fi

CHUNK_ID=$1
NUM_CHUNKS=$2

module load Python/Anaconda_v11.2021

source activate diplom_3

echo "Starting chunk analysis for CHUNK_ID: $CHUNK_ID / NUM_CHUNKS: $NUM_CHUNKS"
echo "Conda environment activated: diplom_3"

/home/mmnima/.conda/envs/diplom_3/bin/python -u /home/mmnima/jokes/graph_maker/item_analyze.py --chunk_id "$CHUNK_ID" --num_chunks "$NUM_CHUNKS"

deactivate

echo "Chunk analysis for CHUNK_ID: $CHUNK_ID finished."