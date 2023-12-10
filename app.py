import os
import random

import torch
import gradio as gr
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


events = [
    "nature retreat",
    "work / office event",
    "wedding as a guest",
    "tropical vacation",
    "conference",
    "sports event",
    "winter vacation",
    "beach",
    "play / concert",
    "picnic",
    "night club",
    "national parks",
    "music festival",
    "job interview",
    "city tour",
    "halloween party",
    "graduation",
    "gala / exhibition opening",
    "fancy date",
    "cruise",
    "casual gathering",
    "concert",
    "cocktail party",
    "casual date",
    "business meeting",
    "camping / hiking",
    "birthday party",
    "bar",
    "business lunch",
    "bachelorette / bachelor party",
    "semi-casual event",
]


def format_instruction(input, context):
    return f"""You are a personal stylist recommending fashion advice and clothing combinations. Use the self body and style description below, combined with the event described in the context to generate 5 self-contained and complete outfit combinations.
        ### Input:
        {input}

        ### Context:
        I'm going to a {context}.

        ### Response:
    """


def main():
    # load base LLM model, LoRA params and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        "neuralwork/mistral-7b-style-instruct",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("neuralwork/mistral-7b-style-instruct")

    def postprocess(outputs, prompt):
        outputs = outputs.detach().cpu().numpy()
        output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        output = output[len(prompt) :]
        return output

    def generate(
        prompt: str,
        event: str,
        top_p: float,
        temperature: float,
        max_new_tokens: int,
        min_new_tokens: int,
        seed: int,
    ):
        torch.manual_seed(seed)
        prompt = format_instruction(str(prompt), str(event))
        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids.cuda()

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
            )

        output = postprocess(outputs, prompt)
        return output

    with gr.Blocks() as demo:
        gr.HTML(
            """
            <h1 style="font-weight: 900; margin-bottom: 7px;">
            Instruct Fine-tune Mistral-7B-v0
            </h1>
            <p>Mistral-7B-v0 fine-tuned on the <a href="https://huggingface.co/datasets/neuralwork/fashion-style-instruct">neuralwork/style-instruct</a> dataset.
            To use the model, simply describe your body type and personal style and select the type of event you're planning to go.
            <br/>
            See our <a href="https://neuralwork.ai/">blog post</a> for a detailed tutorial to fine-tune Mistral on your own dataset.
            <p/>"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    lines=4,
                    label="Style prompt, describe your body type and fashion style.",
                    interactive=True,
                    value="I'm an above average height athletic woman with slightly of broad shoulders and a medium sized bust. I generally prefer a casual but sleek look with dark colors and jeans.",
                )
                event = gr.Dropdown(
                    choices=events, value="semi-casual event", label="Event type"
                )
                seed = gr.Number(
                    value=1371,
                    precision=0,
                    interactive=True,
                    label="Seed for reproducibility, set to -1 to randomize seed",
                )
                top_p = gr.Slider(
                    value=0.9,
                    label="Top p (nucleus sampling)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                )
                max_new_tokens = gr.Slider(
                    minimum=1,
                    maximum=2048,
                    value=1500,
                    label="Maximum new tokens",
                )
                min_new_tokens = gr.Slider(
                    minimum=-1, maximum=2048, value=-1, label="Minimum new tokens"
                )
                temperature = gr.Slider(
                    minimum=0.01, maximum=5, value=0.9, step=0.01, label="Temperature"
                )
                repetition_penalty = gr.Slider(
                    label="Repetition penalty",
                    minimum=1.0,
                    maximum=2.0,
                    step=0.05,
                    value=1.2,
                )
                generate_button = gr.Button("Get outfit suggestions")

            with gr.Column(scale=2):
                response = gr.Textbox(
                    lines=6, label="Outfit suggestions", interactive=False
                )

        gr.Markdown("From [neuralwork](https://neuralwork.ai/) with :heart:")

        generate_button.click(
            fn=generate,
            inputs=[
                prompt,
                event,
                top_p,
                temperature,
                max_new_tokens,
                min_new_tokens,
                seed,
            ],
            outputs=response,
        )

    demo.launch(share=True)


if __name__ == "__main__":
    main()
