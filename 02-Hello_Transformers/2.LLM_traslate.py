from transformers import AutoTokenizer,AutoModelForCausalLM, pipeline

'''
The script loads a pretrained language model, tokenizes an input text prompt for translation, 
and generates a response using the model. It then decodes and prints the generated output.
'''
# -------------  Load model directly ------------------------ #

model_id = "ModelSpace/GemmaX2-28-2B-v0.1"  
tokenizer = AutoTokenizer.from_pretrained(model_id)  # Load the tokenizer for the specified model.
model = AutoModelForCausalLM.from_pretrained(model_id)  # Load the pretrained language model.

# The input text instructs the model to translate the given Chinese sentence into English.
text = "Translate this from Chinese to English:\nChinese: 我爱机器翻译\nEnglish:"  
inputs = tokenizer(text, return_tensors="pt")  # Tokenize the input text and return it as PyTorch tensors.
outputs = model.generate(**inputs, max_new_tokens=50)  # Generate text using the model with a maximum of 50 new tokens.

# Decode the generated tokens back into a human-readable string and print the output.
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))  


# ------------- Use a pipeline as a high-level helper ------------------------ #

# Create a translation pipeline using the specified model
pipeline_translator = pipeline("translation", model="ModelSpace/GemmaX2-28-2B-v0.1")
text = "我爱机器翻译" # Input text in Chinese to be translated
translated_text = pipeline_translator(text) # Perform the translation
#print(translated_text) # Print the translated output


# ------------- Use a pipeline as a high-level helper with another model ------------------------ #

pipeline_translator2 = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
translated_text2 = pipeline_translator2(text)
#print(translated_text2[0]['translation_text'])






