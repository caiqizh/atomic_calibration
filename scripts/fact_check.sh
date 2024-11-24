#!/bin/bash
export OPENAI_API_KEY=""

MODEL_NAMES=("mistral-7b-instruct" "qwen2-7b-instruct" "llama3-8b-instruct")

DATASET_NAMES=("bios" "longfact" "wildhallu")

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
        echo "Fact-checking for model: $MODEL_NAME"
        echo "Fact-checking for dataset: $DATASET_NAME"

        python ../src/fact_check.py \
            --model_name $MODEL_NAME \
            --dataset $DATASET_NAME \
            --num_samples 500 \

    done
    echo "Finished fact-checking for model: $MODEL_NAME on $DATASET_NAME."
done