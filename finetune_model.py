
import os
import argparse

import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


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
    dataset = load_dataset(args.dataset, token=args.auth_token)
    # base model to finetune
    model_id = args.base_model

    # BitsAndBytesConfig to quantize the model int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache=False, device_map="auto")
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # print the number of trainable model params
    print_trainable_parameters(model)

    model_args = TrainingArguments(
        output_dir="mistral-7-style",
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
        disable_tqdm=False
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
        args=model_args,
    )

    # train
    trainer.train() 

    # save model
    trainer.save_model()

    if args.push_to_hub:
        trainer.model.push_to_hub(args.model_name)
        
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="neuralwork/fashion-style-instruct", 
        help="Path to local or HF dataset."
    )
    parser.add_argument(
        "--base_model", type=str, default="mistralai/Mistral-7B-v0.1", 
        help="HF hub id of the base model to finetune."
    )
    parser.add_argument(
        "--model_name", type=str, default="mistral-7b-style-instruct", help="Name of finetuned model."
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
