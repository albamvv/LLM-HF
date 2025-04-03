from transformers import AutoTokenizer,AutoModelForCausalLM, pipeline
import torch



# Load the GPT-2 tokenizer and the GPT-2 language model
tokenizer = AutoTokenizer.from_pretrained("gpt2")  
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2") # Este modelo es autoregresivo, lo que significa que puede predecir el siguiente token basado en el contexto anterior
sentence = "The future of AI is" # Define the input sentence
input_ids = tokenizer(sentence, return_tensors='pt').input_ids  # Tokenize the sentence and convert it into tensor format (PyTorch).
print('input_id: ', input_ids)  

'''
# Loop through each token ID in the tensor and decode it back to a string
for token_id in input_ids[0]:  
    print(tokenizer.decode(token_id))  # Print the corresponding token  
'''
# Pass the tokenized input to the model to obtain output logits, Devuelve logits con las probabilidades de cada token.
logits= gpt2(input_ids).logits
#print('output tensor shape-> ', logits.shape) # Print the shape of the output tensor, which represents model predictions
#print('logits-> ',logits) # Los logits son valores sin procesar que indican la probabilidad de cada token.
final_logits = gpt2(input_ids).logits[0,-1] 
#print('final logits-> ',final_logits)

# NEXT TOKEN
print("Input text->", tokenizer.decode(input_ids[0]))
#print(final_logits.argmax()) # Token ID <--> Index Location Logits
next_token= tokenizer.decode(final_logits.argmax()) 
#print('next token-> ',next_token)

# TOP TEN PREDICTIONS
top_10_logits = torch.topk(final_logits,10)
#for index in top_10_logits.indices:
#    print(tokenizer.decode(index))

# TOP TEN PREDICTIONS PROBABILITY
top10 =torch.topk(final_logits.softmax(dim=0),10)
for value, index in zip(top10.values, top10.indices):
    print(f"{tokenizer.decode(index)} -- {value.item():.1%}")



#next_token = torch.argmax(output.logits[:, -1, :], dim=-1)
#print('next token-> ',tokenizer.decode(next_token))








