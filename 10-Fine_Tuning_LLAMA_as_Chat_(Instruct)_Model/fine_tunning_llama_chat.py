from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import pipeline
from peft import AutoPeftModelForCausalLM
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


#------------------ Normalized quantization --------------------
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", trust_remote_code=True, split="train_sft")
dataset = dataset.shuffle(seed=0).select(range(10_000))
#print(dataset) 
print(dataset[0])

template_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def format_prompt(example):
  """Format the prompt using the <|user|> and <|assistant|> format"""
  chat = example['messages']
  prompt = template_tokenizer.apply_chat_template(chat, tokenize=False)
  return {'text': prompt}

#print(format_prompt(dataset[0])['text'])
#dataset = dataset.map(format_prompt)


#------------------ Testing base LLAMA Model --------------------
# base model
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

pipe = pipeline(task='text-generation', model=model_name, device='cuda')

# prompt
# <|user|>, <|assistant|>

prompt = """<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""

prompt = """
Tell me something about Large Language Models
"""

output = pipe(prompt)
print(output)

#--------------------- Model configuration for training ----------

# do the  4-bit quantization configuration in Q-LORA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = "<PAD>"
tokenizer.padding_size="left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = "auto",
    quantization_config=bnb_config
)

model.config.use_cache=False
model.config.pretraining_tp=1

#--------------------- Prepare LoRA configuration for PEFT Fine tuning ----------

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=64,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

w = 2048*256
a = 2048*64
b = 64*256
w, a, b, a+b, (a+b)/w


#------------ Model fine tuning -----------


output_dir = "train_dir"

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True
)

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
