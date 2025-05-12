from utils import device,prompt1, prompt2,prompt3,prompt4,prompt5,prompt6
from imports import *

'''
The script compares three LLMs: a base model (t5-base), a fine-tuned model (google/flan-t5-base), and a larger fine-tuned model (google/flan-t5-large). 
It loads each model and its tokenizer, tokenizes a given prompt, and generates text. 
The base model produces general text, while the fine-tuned models specialize in tasks like question answering. 
The larger fine-tuned model has more parameters, improving performance. Outputs from each model are printed for comparison. 
The script highlights differences in text generation between pre-trained and fine-tuned models, showing the impact of fine-tuning and model size.
'''

### -------------------  1. Selection of a base pre-trained LLM -------------------------- ################

# This tokenizer is used to convert text into tokens that the model can process
tokenizer_T5 = T5Tokenizer.from_pretrained("t5-base") # Base pre-trained LLM
# Import the pre-trained model
model_T5 = T5ForConditionalGeneration.from_pretrained("t5-base", device_map="auto") # This model has not been fine-tuned for any specific task, it only generates text

# Tokenize the input prompt
prompt_tokens = tokenizer_T5(prompt1, return_tensors="pt").input_ids.to(device)

# Generate text based on the input prompt
# The model continues generating words based on the given prompt
outputs = model_T5.generate(prompt_tokens, max_length=100) # Coge el prompt que se le ha pasado y continua haciendo cosas pero no hace funciones

# Decode the generated tokens into text
print("salida-> ",tokenizer_T5.decode(outputs[0]))


### -------------------  2. Selection of a Fine-tuned LLM -------------------------- ################

# Import the tokenizer for the fine-tuned model
# This tokenizer is specific to the fine-tuned version of the model
tokenizer_FT5 = T5Tokenizer.from_pretrained("google/flan-t5-base") 

# Import the fine-tuned pre-trained model
# This model has been trained for specific tasks like question answering or conversation
model_FT5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")

# Tokenize the prompt
prompt_tokens = tokenizer_T5(prompt1, return_tensors="pt").input_ids.to(device)

# Generate text based on the fine-tuned model
outputs = model_FT5.generate(prompt_tokens, max_length=50)

# Decode and print the generated text
print(tokenizer_FT5.decode(outputs[0]))


### -------------------  3. Selection of a Fine-tuned LLM with 1 billion parameters -------------------------- ################

# Import the tokenizer for the larger fine-tuned model
# This tokenizer is used for the larger version of the fine-tuned model
tokenizer_FT5 = T5Tokenizer.from_pretrained("google/flan-t5-large")

# Import the large fine-tuned model
# This model contains a higher number of parameters, making it more capable
model_FT5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")

# Tokenize the prompt
prompt_tokens = tokenizer_FT5(prompt1, return_tensors="pt").input_ids.to(device)

# Generate text using the large fine-tuned model
outputs = model_FT5.generate(prompt_tokens, max_length=100)

# Decode and print the generated text
print(tokenizer_FT5.decode(outputs[0]))