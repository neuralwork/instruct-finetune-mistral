import os
import argparse
from random import randrange

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def format_instruction(sample):
    return f"""You are a personal stylist recommending fashion advice and clothing combinations. Use the self body and style description below, combined with the event described in the context to generate 5 self-contained and complete outfit combinations.
        ### Input:
        {sample["input"]}

        ### Context:
        {sample["context"]}

        ### Response:
    """

def postprocess(outputs, tokenizer, prompt, sample):
    outputs = outputs.detach().cpu().numpy()
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    output = outputs[0][len(prompt):]

    print(f"Instruction: \n{sample['input']}\n")
    print(f"Context: \n{sample['context']}\n")
    print(f"Ground truth: \n{sample['completion']}\n")
    print(f"Generated output: \n{output}\n\n\n")
    return


def run_model(config):
    # load dataset and select a random sample
    dataset = load_dataset(config.dataset)
    sample = dataset[randrange(len(dataset))]
    prompt = format_instruction(sample)

    # load base LLM model, LoRA params and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        config.model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    
    # inference
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids, 
            max_new_tokens=800, 
            do_sample=True, 
            top_p=0.9,
            temperature=0.9
        )

    postprocess(outputs, tokenizer, prompt, sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="neuralwork/fashion-style-instruct",
        help="HF dataset id or path to local dataset folder."
    )
    parser.add_argument(
        "--model_id", type=str, default="neuralwork/mistral-7b-style-instruct", 
        help="HF LoRA model id or path to local finetuned model folder."
    )

    config = parser.parse_args()
    run_model(config)
