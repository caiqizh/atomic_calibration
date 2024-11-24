
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
import numpy as np
import os
import time 
import os
import json
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import math
import torch
import re


class DIS_vllm:
    def __init__(
        self,
        model: str,
        method: str,
        cuda_devices: str = "0",
        gpu_memory_utilization: float = 0.9,
    ):
        """
        model: str - the name of the model to do discrimination
        method: str - the method to use for the task, either "single" or "context"
        """

        if model == "llama3-8b-instruct":
            model_path = "huggingface/Meta-Llama-3-8B-Instruct"
        elif model == "llama3-70b-instruct":
            model_path = "huggingface/Meta-Llama-3-70B-Instruct"
        elif model == "mistral-7b-instruct":
            model_path = "/apdcephfs_qy3/share_733425/timhuang/cindychung/Mistral-7B-Instruct-v0.2"
        elif model == "mistral-8-7b-instruct":
            model_path = "huggingface/Mixtral-8x7B-Instruct-v0.1"
        elif model == "qwen2-7b-instruct":
            model_path = "huggingface/Qwen2-7B-Instruct"
        elif model == "qwen2-57b-instruct":
            model_path = "huggingface/Qwen2-57B-A14B-Instruct"
        elif model == "qwen2-72b-instruct":
            model_path = "huggingface/Qwen2-72B-Instruct"
        else:
            raise ValueError(f"Model {model} is not supported")
        
        self.model = model
        self.model_path = model_path
        self.method = method
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path, 
            tensor_parallel_size=len(cuda_devices.split(",")),
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048, # For input prompts 
            max_logprobs=10,
            )
        self.sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=0.95,
            max_tokens=100,
            stop_token_ids=[self.tokenizer.eos_token_id],
            skip_special_tokens=True,
            logprobs=10,
        )

        self.template_single = (
            "Please evaluate the truthfulness of the following statement based on your internal knowledge. Respond only with 'True' or 'False'—no other information.\n\n"
            "Statement: {atomic_fact_piece}\n"
            "True or False: "
        )

        self.template_single_icl = (
            "Please evaluate the truthfulness of the following statement based on your internal knowledge. Respond only with 'True' or 'False'—no other information.\n\n"
            "For example:\n"
            "Statement: The earth revolves around the sun.\n"
            "True or False: True\n\n"
            "Now, rate the following statement:\n"
            "Statement: {atomic_fact_piece}\n"
            "True or False: "
        )

        self.template_context = (
            "You are given a topic and a statement related to that topic. Evaluate the truthfulness of the statement based solely on your internal knowledge. A statement is true only if it is factually accurate according to what you know.\n\n"
            "Do not use the topic to evaluate the statement. Respond with 'True' or 'False' only, without any additional information.\n\n"
            "Topic: {context}\n"
            "Statement: {atomic_fact_piece}\n"
            "True or False: "
        )


        self.template_context_icl = (
            "You are given a topic and a statement related to that topic. Evaluate the truthfulness of the statement based solely on your internal knowledge. A statement is considered true only if it is factually accurate according to what you know.\n\n"
            "Do not use the topic to evaluate the statement. Respond with 'True' or 'False' only, without adding any additional information.\n\n"
            "For example:\n"
            "Topic: [Sample topic]\n"
            "Statement: [Sample statement]\n"
            "True or False: True\n\n"
            "Now, evaluate the following statement:\n"
            "Topic: {context}\n"
            "Statement: {atomic_fact_piece}\n"
            "True or False: "
        )


        self.template_rating = (
            "Please rate your confidence in the truth of the following statement on a scale from 0 to 10.\n"
            "A confidence of 0 means you are not confident at all whether the statement is true, while 10 means you are absolutely confident the statement is true.\n"
            "The topic provides context for the statement, but focus on the statement itself when rating your confidence.\n"
            "Provide only the numerical value for your confidence level, bounded by dollar signs ($), without any additional information.\n\n"
            "Topic: {context}\n"
            "Statement: {atomic_fact_piece}\n"
            "Confidence: "
        )


        self.template_rating_icl = (
            "Please rate your confidence in the truth of the following statement on a scale from 0 to 10.\n"
            "A confidence of 0 means you are not confident at all whether the statement is true, while 10 means you are absolutely confident the statement is true.\n"
            "The topic provides context for the statement, but your confidence rating should focus on the statement itself.\n"
            "Provide only the numerical value for your confidence level, bounded by dollar signs ($), without any additional information.\n\n"
            "For example:\n"
            "Topic: Tell me about the solar system\n"
            "Statement: The earth revolves around the sun.\n"
            "Confidence: $10$\n\n"
            "Now, rate the following statement:\n"
            "Topic: {context}\n"
            "Statement: {atomic_fact_piece}\n"
            "Confidence: "
        )
        
        self.not_defined_text = set()

    def get_prompt(self, original_prompt):
        if "mistral" in self.model:
            messages = [
                {"role": "user", "content": original_prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": original_prompt}
            ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return prompt
    
    def completion(self, prompts: str):
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        return outputs

    def clean_token(self, token):
        """ Clean the token by stripping spaces, parentheses, and converting to lowercase. """
        return re.sub(r'[()\s]', '', token).lower()

    def extract_confidence_score(self, generate_text):
        # Use regex to find the number prefixed by a dollar sign, optionally followed by a dollar sign or punctuation
        match = re.search(r"\$(\d+)(?=\$|[,.\s]|$)", generate_text)
        
        if match:
            score = int(match.group(1))
            return score / 10
        else:
            print(f"No valid confidence score found in output: {generate_text}")
            return -1


    def get_p_true(self, logprobs):
        if not logprobs:
            return 0
        
        # Step 1: Store the original tokens and their logprobs
        original_token_prob = {}
        for item in logprobs:
            logprobs_obj = logprobs[item]
            logprob = -float('inf') if str(logprobs_obj.logprob) == '-inf' else float(logprobs_obj.logprob)
            original_token_prob[logprobs_obj.decoded_token] = logprob

        # Convert the original log probabilities to a tensor
        original_tokens = list(original_token_prob.keys())
        logprob_tensor = torch.tensor(list(original_token_prob.values()), dtype=torch.float32)
        
        # Step 2: Apply softmax to the original tokens
        softmax_probs = torch.softmax(logprob_tensor, dim=-1)
        
        # Create a dictionary of original tokens with their softmax probabilities
        original_softmax_prob_dict = {token: prob.item() for token, prob in zip(original_tokens, softmax_probs)}
        
        # Step 3: Apply the cleaning function to the tokens and combine their probabilities
        cleaned_prob_dict = {}
        for token, prob in original_softmax_prob_dict.items():
            cleaned_token = self.clean_token(token)
            if cleaned_token in cleaned_prob_dict:
                cleaned_prob_dict[cleaned_token] += prob
            else:
                cleaned_prob_dict[cleaned_token] = prob
        
        # Step 4: Return the probability of "true" (cleaned version of "True")
        return cleaned_prob_dict.get("true", 0)
        

    def predict(
        self,
        context: str,
        targets: str,
    ):
        '''
        contexts: List[str] - the context to evaluate the atomic facts
        targets: List[str] - the atomic facts to evaluate
        '''

        num_atomic_facts = len(targets)
        scores = np.zeros((num_atomic_facts))
        prompts = []

        for target in targets:
            if self.method == "single":
                prompt_text = self.template_single_icl.format(atomic_fact_piece=target)
            elif self.method == "context":
                prompt_text = self.template_context_icl.format(context=context, atomic_fact_piece=target)
            elif self.method == "rating":
                prompt_text = self.template_rating_icl.format(context=context, atomic_fact_piece=target)
            else:
                raise ValueError(f"Method {self.method} is not supported")

            # print(prompt_text)
            prompt = self.get_prompt(prompt_text)

            prompts.append(prompt)

        outputs = self.completion(prompts)

        if self.method == "rating":
            for target_i, output in enumerate(outputs):
                generate_text = output.outputs[0].text
                # print(generate_text)
                score = self.extract_confidence_score(generate_text)
                scores[target_i] = score

            scores_per_sentence = scores
            return scores_per_sentence, scores
        else:
            for target_i, output in enumerate(outputs):
                generate_text = output.outputs[0].text
                print(output.outputs[0].logprobs[0])
                p_true = self.get_p_true(output.outputs[0].logprobs[0])
                scores[target_i] = p_true
            scores_per_sentence = scores
        return scores_per_sentence, scores


if __name__ == "__main__":

    os.environ['HF_HOME'] = 'huggingface'
    os.environ['HUGGINGFACE_HUB_CACHE'] = 'huggingface'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    dis_vllm = DIS_vllm(model="qwen2-7b-instruct", method="rating", gpu_memory_utilization=0.9, cuda_devices="0,1,2,3")

    atomic_facts = ['Donald Trump is the 45th president of the United States.', 'Donald Trump was born on June 14, 1946, in Queens, New York City.', 'Donald Trump was born in Los Angeles, California.', 'Donald Trump received a Bachelor of Science in economics from the University of Pennsylvania in 1968.', 'Donald Trump launched side ventures, mostly licensing the Trump name.', 'Donald Trump co-produced and hosted the reality television series The Apprentice from 2004 to 2015.', 'Donald Trump and his businesses have been plaintiffs or defendants in more than 4,000 legal actions, including six business bankruptcies.']

    # atomic_facts = ['University of Cambridge is located in Cambridge, England.', 'University of Cambridge is located in Cambridge, Massachusetts.', 'University of Cambridge is located in Cambridge, Ontario.', 'University of Cambridge is located in Cambridge, New York.', 'University of Cambridge is located in Cambridge, Maryland.', 'University of Cambridge is located in Cambridge, Idaho.', 'University of Cambridge is located in Cambridge, Illinois.']

    topic = 'Donald John Trump' 

    answer = "The University of Cambridge is a public collegiate research university in Cambridge, England. Founded in 1209, the University of Cambridge is the world's third-oldest university in continuous operation. The university's founding followed the arrival of scholars who left the University of Oxford for Cambridge after a dispute with local townspeople.[8][9] The two ancient English universities, although sometimes described as rivals, share many common features and are often jointly referred to as Oxbridge."

    scores = dis_vllm.predict(topic, atomic_facts)
    print(scores)