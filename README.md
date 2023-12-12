## Fine-tuning LLMs with PEFT
This project is a tutorial on parameter-efficient fine-tuning (PEFT) and quantization of the [Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) model. We use LoRA for PEFT and 4-bit quantization to compress the model, and fine-tune the model on a semi-manually crafted fashion style recommendation instruct [dataset](https://huggingface.co/datasets/neuralwork/fashion-style-instruct). For more information and a step by step guide, see our [blog post](https://neuralwork.ai/).

## Usage
Start by cloning the repository, setting up a conda environment and installing the dependencies. We tested our scripts with python 3.9 and CUDA 11.7.
```
git clone https://github.com/neuralwork/finetune-mistral.git
cd finetune-mistral

conda create -n llm python=3.9
conda activate llm
pip install -r requirements.txt
```

You can finetune the model on our fashion-style-instruct [dataset](https://huggingface.co/datasets/neuralwork/fashion-style-instruct) or another dataset. Note that you will need to have the same features as our dataset and pass in your HF Hub token as an argument if using a private dataset. Fine-tuning takes about 2 hours on a single A40, you can either use the default accelerate settings or configure it to use multiple GPUS. To fine-tune the model:
```
accelerate config default

python finetune_model.py --dataset=<HF_DATASET_ID_OR_PATH> --base_model="NousResearch/Llama-2-7b-hf" --model_name=<YOUR_MODEL_NAME> --auth_token=<HF_AUTH_TOKEN> --push_to_hub
```

One model training is completed, only the fine-tuned (LoRA) parameters are saved, which are loaded to overwrite the corresponding parameters of the base model during testing.  

To test the fine-tuned model with a random sample selected from the dataset, run `python test.py`. To launch the full Gradio demo and play around with your own examples, launch the demo with `python app.py`


## License
This project is licensed under the [MIT license](https://github.com/neuralwork/finetune-mistral/blob/main/LICENSE).

From [neuralwork](https://neuralwork.ai/) with :heart:
