#!/bin/bash

export OPENAI_API_KEY=""

MODEL_NAMES=("mistral-7b-instruct" "qwen2-7b-instruct" "llama3-8b-instruct")
MODEL_NAMES=("qwen2-57b-instruct" "mistral-8-7b-instruct" "llama3-70b-instruct" "qwen2-72b-instruct")
DATASET_NAMES=("bios" "longfact" "wildhallu")

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET_NAME in "${DATASET_NAMES[@]}"; do
        echo "Generating atomic facts for model: $MODEL_NAME"
        echo "Generating atomic facts for dataset: $DATASET_NAME"

        python ../src/generate_atomic_facts.py \
            --model_name $MODEL_NAME \
            --dataset $DATASET_NAME \

    done
    echo "Finished generating atomic facts for model: $MODEL_NAME on $DATASET_NAME."
done