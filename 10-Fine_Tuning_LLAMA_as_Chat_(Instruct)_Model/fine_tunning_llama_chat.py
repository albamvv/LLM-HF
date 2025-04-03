from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import pipeline
import torch
from config import bnb_config, model, model_name, LoraConfig
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


#------------------ Normalized quantization --------------------
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", trust_remote_code=True, split="train_sft")
dataset = dataset.shuffle(seed=0).select(range(10_000))
#print(dataset) #print(dataset[0])

template_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
#print(template_tokenizer)

def format_prompt(example):
  """Format the prompt using the <|user|> and <|assistant|> format"""
  chat = example['messages']
  prompt = template_tokenizer.apply_chat_template(chat, tokenize=False)
  return {'text': prompt}
#print(format_prompt(dataset[0])['text'])
dataset = dataset.map(format_prompt)
#print(dataset[0])


#------------------ Testing base LLAMA Model --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipeline(task='text-generation', model=model_name, device=device)

# prompt
# <|user|>, <|assistant|>

prompt = """<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""

'''
prompt = """
Tell me something about Large Language Models
"""
'''
output = pipe(prompt)
print(output)
#--------------------- Model configuration for training ----------

# do the  4-bit quantization configuration in Q-LORA
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = "<PAD>"
tokenizer.padding_size="left"

model.config.use_cache=False
model.config.pretraining_tp=1

#--------------------- Prepare LoRA configuration for PEFT Fine tuning ----------
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

w = 2048*256
a = 2048*64
b = 64*256
w, a, b, a+b, (a+b)/w


#------------ Model fine tuning -----------
trainer = SFTTrainer(
    model=model,
    train_dataset = dataset,
    dataset_text_field='text',
    tokenizer = tokenizer,
    args=args,
    max_seq_length=512,
    peft_config = peft_config
)
trainer.train()
trainer.model.save_pretrained("TinyLlama-1.1B-qlora")

#--------- Load Pre-Trained PEFT Model for Prediction

model = AutoPeftModelForCausalLM.from_pretrained(
    "TinyLlama-1.1B-qlora",
    device_map='auto'
)

merged_model = model.merge_and_unload()

prompt = """<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = "<PAD>"
tokenizer.padding_size="left"

pipe = pipeline(task='text-generation', model=merged_model, tokenizer=tokenizer)
output = pipe(prompt)
print(output[0]['generated_text'])

#!zip -r tiny_llama_qlora_adapter.zip TinyLlama-1.1B-qlora

