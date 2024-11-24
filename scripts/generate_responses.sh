#!/bin/bash

MODEL_NAMES=("mistral-7b-instruct" "qwen2-7b-instruct" "llama3-8b-instruct")
DATASET_NAMES=("bios" "longfact" "wildhallu")

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET_NAME in "${DATASET_NAMES[@]}"; do
        echo "Processing model: $MODEL_NAME"
        echo "Processing dataset: $DATASET_NAME"

        python ../src/generate_responses_vllm.py \
            --cuda_devices "4,5,6,7" \
            --gpu_memory_utilization 0.8 \
            --dataset $DATASET_NAME \
            --model_name $MODEL_NAME \
            --generate_answers \

        sleep 10
    done
done

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET_NAME in "${DATASET_NAMES[@]}"; do
        echo "Processing model: $MODEL_NAME"
        echo "Processing dataset: $DATASET_NAME"

        python ../src/generate_responses_vllm.py \
            --cuda_devices "4,5,6,7" \
            --gpu_memory_utilization 0.9 \
            --dataset $DATASET_NAME \
            --model_name $MODEL_NAME \
            --generate_samples \

        sleep 10
    done
done