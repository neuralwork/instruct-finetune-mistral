
import os
import argparse

import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


def format_instruction(sample):
    return f"""You are a personal stylist recommending fashion advice and clothing combinations. Use the self body and style description below, combined with the event described in the context to generate 5 self-contained and complete outfit combinations.
        ### Input:
        {sample["input"]}

        ### Context:
        {sample["context"]}

        ### Response:
        {sample["completion"]}
    """

def finetune_model(args):
    dataset = load_dataset(args.dataset, auth_token=args.auth_token)
    # base model to finetune
    model_id = args.base_model

    # BitsAndBytesConfig to quantize the model int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache=False, device_map="auto")
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM")

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    args = TrainingArguments(
        output_dir="llama-7-int4-style",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=True
    )

    max_seq_length = 2048

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_instruction,
        args=args,
    )

    # train
    trainer.train() 

    # save model
    trainer.save_model()

    if args.push_to_hub:
        trainer.model.push_to_hub(args.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="neuralwork/fashion-style-instruct", 
        help="Path to local or HF dataset."
    )
    parser.add_argument(
        "--base_model", type=str, default="NousResearch/Llama-2-7b-hf", 
        help="HF hub id of the base model to finetune."
    )
    parser.add_argument(
        "--model_name", type=str, default="llama-2-4bit", 
        help="Name of finetuned model, only used if push_to_hub option is passed."
    )
    parser.add_argument(
        "--auth_token", type=str, default=None, 
        help="HF authentication token, only used if downloading a private dataset."
    )
    parser.add_argument(
        "--push_to_hub", default=False, action="store_true", 
        help="Whether to push finetuned model to HF hub."
    )
    args = parser.parse_args()
    finetune_model(args)