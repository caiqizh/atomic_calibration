import json
from openai import OpenAI
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
import sys
sys.path.append('../utils')
from factchecker import FactChecker
from abstrain_detection import is_response_abstained

# Read jsonl file
def read_jsonl(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main(args):
    # Load the data
    data = read_jsonl(f"../results/{args.dataset}/{args.model_name}/{args.model_name}_atomic_facts.jsonl")
    
    # Check if the file already exists
    previous_file = f"../results/{args.dataset}/{args.model_name}/{args.model_name}_atomic_facts_veracity.jsonl"
    if os.path.exists(previous_file):
        data_to_save = read_jsonl(previous_file)
    else:
        data_to_save = []

    # Remove the items that are already processed, start from where we left off
    data = [item for item in data if item["prompt"] not in [x["prompt"] for x in data_to_save]]
    print("Loaded {} items from previous file".format(len(data_to_save)))
    
    if args.debug:
        data = data[:1]
    
    # TODO: Consider removing this line in the future
    # Only use the first num_samples
    data = data[:args.num_samples]

    fc = FactChecker(max_evidence_length=10000)

    for item in tqdm(data, desc=args.model_name):
        if "atomic_facts_veracity" in item:
            print("Already processed: ", item["prompt"])
            continue
        else:
            topic = item["topic"]
            atomic_facts = item["atomic_facts"]

            if is_response_abstained(item["answer"], "generic"):
                item["is_abstained"] = True
                print("Abstained: ", item["prompt"])
                gpt_labels = []
            else:
                item["is_abstained"] = False
                gpt_labels, raw_completion = fc.get_veracity_labels(topic=topic, 
                                                 atomic_facts=atomic_facts, 
                                                 knowledge_source=args.dataset, 
                                                 evidence_type="zero")

            item["atomic_facts_veracity"] = gpt_labels
            item["raw_completion"] = raw_completion
            data_to_save.append(item)

            output_dir = f"../results/{args.dataset}/{args.model_name}/{args.model_name}_atomic_facts_veracity.jsonl"
            with open(output_dir, "w") as f:
                for item in data_to_save:
                    f.write(json.dumps(item) + "\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using LLM")
    parser.add_argument("--dataset", type=str, default="bios", help="Dataset for LLM")
    parser.add_argument("--model_name", type=str, default="llama3-8b", help="Model ID for LLM")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to fact-check.")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode")

    args = parser.parse_args()
    main(args)

'''
python fact_check.py --dataset longfact --model_name llama3-8b-instruct --debug
'''