
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

class LUQ_vllm:

    def __init__(
        self,
        nli_model: str,
        method: str,
        cuda_devices: str = "0",
        gpu_memory_utilization: float = 0.9,
    ):
        """
        nli_model: str - the name of the model to do NLI
        method: str - the method to use for the task, either "binary" or "multiclass"
        """

        if nli_model == "llama3-8b-instruct":
            model_path = "huggingface/Meta-Llama-3-8B-Instruct"
        else:
            raise ValueError("Model not supported")

        self.method = method
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=0.9,
            max_tokens=5,
            stop_token_ids=[self.tokenizer.eos_token_id],
            skip_special_tokens=True,
        )

        self.llm = LLM(
            model=model_path, 
            tensor_parallel_size=len(cuda_devices.split(",")),
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048, # For input prompts 
            )
        
        if self.method == "binary":
            self.prompt_template = (
                "Context: {context}\n\n"
                "Sentence: {sentence}\n\n"
                "Is the sentence supported by the context above?\n\n"
                "You should answer the question purely based on the given context and not your own knowledge. "
                "Do not output the explanations.\n\n"
                "Your answer should be within \"yes\" or \"no\".\n\n"
                "Answer: "
            )
            self.text_mapping = {'yes': 1, 'no': 0, 'n/a': 0.5}

        elif self.method == "multiclass":
            self.prompt_template = (
                "Context: {context}\n\n"
                "Sentence: {sentence}\n\n"
                "Is the sentence supported, refuted or not mentioned by the context above?\n\n"
                "You should answer the question purely based on the given context and not your own knowledge. "
                "Do not output the explanations.\n\n"
                "Your answer should be within \"supported\", \"refuted\", or \"not mentioned\".\n\n"
                "Answer: "
            )
            self.text_mapping = {'supported': 1, 'refuted': 0, 'not mentioned': -1, 'n/a': 0.5}

        self.not_defined_text = set()

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template
    
    def completion(self, prompts: str):
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        return outputs

    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        raw_scores = np.zeros((num_sentences, num_samples))
        
        for sent_i in range(num_sentences):
            prompts = []
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                # this seems to improve performance when using the simple prompt template
                sample = sample.replace("\n", " ") 
                prompt_text = self.prompt_template.format(context=sample, sentence=sentence)
                # print(prompt_text)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_text}
                ]

                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

                prompts.append(prompt)

            outputs = self.completion(prompts)

            for sample_i, output in enumerate(outputs):
                generate_text = output.outputs[0].text
                # print(generate_text)
                score_ = self.text_postprocessing(generate_text)
                # print(score_)
                raw_scores[sent_i, sample_i] = score_

        scores_filtered = np.ma.masked_equal(raw_scores, -1)
        scores_per_sentence = scores_filtered.mean(axis=-1)
        scores_per_sentence = np.where(scores_per_sentence.mask, 0, scores_per_sentence)
        # print(scores_per_sentence)
        return scores_per_sentence, raw_scores
        

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        """

        if self.method == "binary":
            text = text.lower().strip()
            if text[:3] == 'yes':
                text = 'yes'
            elif text[:2] == 'no':
                text = 'no'
            else:
                if text not in self.not_defined_text:
                    print(f"warning: {text} not defined")
                    self.not_defined_text.add(text)
                text = 'n/a'
            return self.text_mapping[text]
        
        elif self.method == "multiclass":
            text = text.lower().strip()
            if text[:7] == 'support':
                text = 'supported'
            elif text[:5] == 'refut':
                text = 'refuted'
            elif text[:3] == 'not':
                text = 'not mentioned'
            else:
                if text not in self.not_defined_text:
                    print(f"warning: {text} not defined")
                    self.not_defined_text.add(text)
                text = 'n/a'
            return self.text_mapping[text]
            

if __name__ == "__main__":

    os.environ['HF_HOME'] = 'huggingface'
    os.environ['HF_HUB_CACHE'] = 'huggingface'
    os.environ["HF_TOKEN"] = "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"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

    LUQ_vllm = LUQ_vllm(nli_model = "llama3-8b-instruct", method = "multiclass",
    cuda_devices="4,5,6,7", gpu_memory_utilization=0.9)

    sentences = ['Michael Alan Weiner (born March 31, 1942) is an American radio host.', 'Michael Alan Weiner is the host of The Savage Nation.', "Michael Alan Weiner was a good student when he was in American."]

    # Other samples generated by the same LLM to perform self-check for consistency
    sample1 = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is the host of The Savage Country."
    sample2 = "Michael Alan Weiner (born January 13, 1960) is a Canadian radio host. He works at The New York Times."
    sample3 = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He obtained his PhD from MIT."

    luc_scores = LUQ_vllm.predict(sentences, [sample1, sample2, sample3])

    print(luc_scores)