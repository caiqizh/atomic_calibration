import os
import json
from tqdm import tqdm
import argparse
from openai import OpenAI
import time
from transformers import GPT2Tokenizer


def estimate_cost(input_tokens, output_tokens):
    # Cost per million tokens for gpt-4-1106-preview model
    INPUT_COST_PER_MILLION = 5.00  # USD
    OUTPUT_COST_PER_MILLION = 15.00  # USD
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
    total_cost = input_cost + output_cost
    return total_cost


def estimate_overall_cost(data):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    total_input_tokens = sum(len(tokenizer.encode(item["answer"])) for item in data)
    # Assume the output is twice the input; As we are asking for atomic facts, the output should be more than the input
    total_output_tokens = total_input_tokens
    total_cost = estimate_cost(total_input_tokens, total_output_tokens)
    return total_cost


def get_completion(user_prompt, retries=100, delay=2):
    client = OpenAI(
        base_url="https://gptproxy.llmpaas.woa.com/v1", 
        api_key=os.getenv("OPENAI_API_KEY")
    )

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return None


prompt_template = """
Please breakdown the following passage into independent fact pieces. 

Step 1: For each sentence, you should break it into several fact pieces. Each fact piece should only contain one single independent fact. Normally the format of a fact piece is "subject + verb + object". If the sentence does not contain a verb, you can use "be" as the verb.

Step 2: Do this for all the sentences. Output each piece of fact in one single line starting with ###. Do not include other formatting. 

Step 3: Each atomic fact should be self-contained. Do not use pronouns as the subject of a piece of fact, such as he, she, it, this that, use the original subject whenever possible.

Here are some examples:

Example 1:
Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969.
### Michael Collins was born on October 31, 1930.
### Michael Collins is retired.
### Michael Collins is an American.
### Michael Collins was an astronaut.
### Michael Collins was a test pilot.
### Michael Collins was the Command Module Pilot.
### Michael Collins was the Command Module Pilot for the Apollo 11 mission.
### Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969.

Example 2:
League of Legends (often abbreviated as LoL) is a multiplayer online battle arena (MOBA) video game developed and published by Riot Games. 
### League of Legends is a video game.
### League of Legends is often abbreviated as LoL.
### League of Legends is a multiplayer online battle arena.
### League of Legends is a MOBA video game.
### League of Legends is developed by Riot Games.
### League of Legends is published by Riot Games.

Example 3:
Emory University has a strong athletics program, competing in the National Collegiate Athletic Association (NCAA) Division I Atlantic Coast Conference (ACC). The university's mascot is the Eagle.
### Emory University has a strong athletics program.
### Emory University competes in the National Collegiate Athletic Association Division I.
### Emory University competes in the Atlantic Coast Conference.
### Emory University is part of the ACC.
### Emory University's mascot is the Eagle.

Now it's your turn. Here is the passage: 

{}

You should only return the final answer. Now your answer is:
"""

def get_atomic_facts(args, generation_file_name):
    data = []
    with open(generation_file_name, "r") as f:
        for line in f:
            data.append(json.loads(line))

    if args.debug:
        data = data[:2]

    estimated_cost = estimate_overall_cost(data)
    print(f"Estimated overall cost: ${estimated_cost:.6f}")

    output_file_name = f"../results/{args.dataset}/{args.model_name}/{args.model_name}_atomic_facts.jsonl"
    
    if os.path.exists(output_file_name):
        with open(output_file_name, "r") as f:
            processed_items = [json.loads(line) for line in f.readlines()]
        
        # Filter out items that have already been processed
        data = [item for item in data if item["prompt"] not in {processed_item["prompt"] for processed_item in processed_items}]
    
    print(f"Continue from {len(data)} items")

    new_data = []
    for item in tqdm(data, desc=args.model_name):
        answer = item["answer"]
        # print(answer)
        try:
            raw_atomic_facts = get_completion(prompt_template.format(answer))
            atomic_facts = [fact.strip() for fact in raw_atomic_facts.split("###") if fact.strip()]
            # print(raw_atomic_facts)
        except Exception as e:
            print(f"Error: {e}. Skipping...")
            atomic_facts = []
            raw_atomic_facts = []

        new_item = {
            "prompt": item["prompt"],
            "answer": item["answer"],
            'topic': item['topic'],
            "atomic_facts": atomic_facts,
            "raw_atomic_facts": raw_atomic_facts
        }
        new_data.append(new_item)

        with open(output_file_name, "a") as f:
            f.write(json.dumps(new_item) + "\n")
    

def main(args):
    if args.dataset == "bios":
        generation_file_name = f"../results/bios/{args.model_name}/{args.model_name}_answers.jsonl"
        get_atomic_facts(args, generation_file_name)
    elif args.dataset == "longfact":
        generation_file_name = f"../results/longfact/{args.model_name}/{args.model_name}_answers.jsonl"
        get_atomic_facts(args, generation_file_name)
    elif args.dataset == "wildhallu":
        generation_file_name = f"../results/wildhallu/{args.model_name}/{args.model_name}_answers.jsonl"
        get_atomic_facts(args, generation_file_name)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using LLM")
    parser.add_argument("--dataset", type=str, default="bios", help="Dataset for LLM")
    parser.add_argument("--model_name", type=str, default="llama3-8b-instruct", help="Model name")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode")
    args = parser.parse_args()
    main(args)

'''
export OPENAI_API_KEY=""

python generate_atomic_facts.py --model_name llama3-8b-instruct --dataset bios --debug
'''