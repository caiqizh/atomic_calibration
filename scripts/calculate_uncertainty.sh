#!/bin/bash
MODEL_NAMES=("mistral-7b-instruct" "qwen2-7b-instruct" "llama3-8b-instruct")
DATASET_NAMES=("bios" "longfact" "wildhallu")
confidence_types=("generative" "discriminative")
generative_methods=("binary" "multiclass")
discriminative_methods=("single" "context" "rating")
discriminative_methods=("single" "context")


##################### For debugging #####################
# MODEL_NAMES=("qwen2-7b-instruct")
# DATASET_NAMES=("bios")
# confidence_types=("discriminative")
# generative_methods=("binary")
# discriminative_methods=("context")
#########################################################

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
        echo "Calculating uncertainty for model: $MODEL_NAME"
        echo "Calculating uncertainty for dataset: $DATASET_NAME"

        # Loop through each confidence type and method
        for confidence_type in "${confidence_types[@]}"; do
            if [ "$confidence_type" == "generative" ]; then
                for method in "${generative_methods[@]}"; do
                    echo "Running $confidence_type with $method method"
                    python ../src/calculate_uncertainty.py \
                        --confidence_type $confidence_type \
                        --confidence_method $method \
                        --model_name $MODEL_NAME \
                        --dataset $DATASET_NAME \
                        --cuda_devices 0,1,2,3,4,5,6,7 \
                        --gpu_memory_utilization 0.9 \
                        --overwrite \
                        # --debug
                done
            elif [ "$confidence_type" == "discriminative" ]; then
                for method in "${discriminative_methods[@]}"; do
                    echo "Running $confidence_type with $method method"
                    python ../src/calculate_uncertainty.py \
                        --confidence_type $confidence_type \
                        --confidence_method $method \
                        --model_name $MODEL_NAME \
                        --dataset $DATASET_NAME \
                        --cuda_devices 0,1,2,3,4,5,6,7 \
                        --gpu_memory_utilization 0.9 \
                        --overwrite \
                        # --debug
                done
            fi
        done
    done
    echo "Finished calculating uncertainty for model: $MODEL_NAME on $DATASET_NAME."
done