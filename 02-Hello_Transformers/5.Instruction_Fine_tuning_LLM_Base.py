from imports import *
from utils import device,text1, text2,text3
# Download necessary NLP resources
nltk.download("punkt")


### ------------------------ 1. Behavior of Flan-T5-small without Fine-tuning -------------------------------- ########
# Load the tokenizer for Flan-T5-small model
tokenizer_FT5 = T5Tokenizer.from_pretrained("google/flan-t5-small")

# Load the pre-trained model with automatic device mapping
model_FT5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto")

# Define a prompt template for text summarization
prompt_template = f"Resume el siguiente articulo:\n\n{text3}"
#prompt_template = "translate English to German: How old are you?"

# Tokenize the prompt 
prompt_tokens = tokenizer_FT5(prompt_template, return_tensors="pt").input_ids.to(device)

# Generate output tokens based on the model's prediction
outputs = model_FT5.generate(prompt_tokens, max_length=200)

# Decode and print the generated text
print("Generated text-> ",tokenizer_FT5.decode(outputs[0]))


'''

### ------------------------ 2. Selection and preparation of the dataset -------------------------------- ########

# Load the MLSUM dataset in Spanish
ds = load_dataset("mlsum", 'es',trust_remote_code=True)
#print(ds)
# Mostramos un ejemplo del subconjunto de datos de entrenamiento
#print(ds["train"]["text"][10]) #Muestra el décimo artículo de la sección de entrenamiento.
# Mostramos el resumen correspondiente al ejemplo anterior
#print(ds["train"]["summary"][10]) # Muestra el resumen correspondiente al décimo artículo.

# Reduce dataset size for training, validation, and testing
NUM_EJ_TRAIN = 1500
NUM_EJ_VAL = 500
NUM_EJ_TEST = 200

ds['train'] = ds['train'].select(range(NUM_EJ_TRAIN))
ds['validation'] = ds['validation'].select(range(NUM_EJ_VAL))
ds['test'] = ds['test'].select(range(NUM_EJ_TEST))

#print("ds-->  ",ds)
# features: ['text', 'summary', 'topic', 'url', 'title', 'date'],

# Preprocessing function to format dataset entries
def parse_dataset(ejemplo):
  """Procesa los ejemplos para adaptarlos a la plantilla."""
  return {"prompt": f"Resume el siguiente articulo:\n\n{ejemplo['text']}"}

ds["train"] = ds["train"].map(parse_dataset)
ds["validation"] = ds["validation"].map(parse_dataset)
ds["test"] = ds["test"].map(parse_dataset)

#print("ds-->  ",ds)
#features: ['text', 'summary', 'topic', 'url', 'title', 'date', 'prompt']

'''
#print("-----------------------------")
#print(ds["train"]["prompt"][10])
#print("-----------------------------")
#print(ds["train"]["text"][10])
'''

### ------------------------ 3. Tokenization of the dataset -------------------------------- ########

# Tokenize the prompts and truncate them if needed
#Paso 1: Junta (concatena) los datos de entrenamiento, validación y prueba en un solo conjunto de datos.
#Paso 2: Tokeniza los textos de la columna "prompt" y los trunca si son demasiado largos.
prompts_tokens = concatenate_datasets([ds["train"], ds["validation"], ds["test"]]).map(lambda x: tokenizer(x["prompt"], truncation=True), batched=True) # Va a truncar en 512 que es el tamaño máximo para este modelo
#print('prompts_tokens--> ',prompts_tokens) # features: ['text', 'summary', 'topic', 'url', 'title', 'date', 'prompt', 'input_ids', 'attention_mask']
# input_ids → Son los números que representan cada palabra/token

# Compute the maximum token length for prompts
max_token_len = max([len(x) for x in prompts_tokens["input_ids"]])
#print(f"Maximo tamaño de prompt: {max_token_len}")

# Tokenize the summaries (targets) and truncate them
completions_tokens = concatenate_datasets([ds["train"], ds["validation"], ds["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True)
#print("completion tokens--> ",completions_tokens) # features: ['text', 'summary', 'topic', 'url', 'title', 'date', 'prompt', 'input_ids', 'attention_mask']

# Compute the maximum token length for summaries
max_completion_len = max([len(x) for x in completions_tokens["input_ids"]])
#print(f"Maximo tamaño de completion: {max_completion_len}")

# Function to tokenize and format dataset with padding
def padding_tokenizer(datos):
    """
    Tokenizes prompts and summaries with padding and truncation,
    ensuring all have the same size. Padding tokens are replaced with -100
    so the model ignores them during training.
    """
    model_inputs = tokenizer(datos['prompt'], max_length=max_token_len, padding="max_length", truncation=True)
    model_labels = tokenizer(datos['summary'], max_length=max_completion_len, padding="max_length", truncation=True)
    #print("model labels--> ",model_labels)

    # Replace padding tokens with -100 for the labels
    model_labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_labels["input_ids"]]
    model_inputs['labels'] = model_labels["input_ids"]

    return model_inputs

ds_tokens = ds.map(padding_tokenizer, batched=True, remove_columns=['text', 'summary', 'topic', 'url', 'title', 'date', 'prompt'])
'''
    #'input_ids': [3, 456, 789, 1, 0, 0, 0, 0, 0, 0],  # Prompt tokenizado
   # 'attention_mask': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0], # Máscara de atención
   # 'labels': [3, 678, 890, 1, -100, -100, -100, -100, -100, -100] # Summary tokenizado con -100 en padding
'''
#print(ds_tokens)
#print("input ids--> ",ds_tokens["train"]["input_ids"][10])
#print("labels--> ",ds_tokens["train"]["labels"][10])


### ------------------------ 4. Model Fine-tuning -------------------------------- ########

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Load ROUGE metric for evaluation
metric = evaluate.load("rouge")

# Funciona auxiliar para preprocesar el texto
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum espera una nueva línea después de cada frase
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Reemplazamos -100 en las etiquetas porque no podemos decodificarlo
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Preprocesamos el texto
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


#3. Lectura y adaptación de los datos para el entrenamiento

# Ignoramos los tokens relacionados con el padding durante el proceso de entrenamiento para los prompts
label_pad_token_id = -100

# Data collator to manage batch processing during training
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

#4. Preparación y ejecución del fine-tuning (entrenamiento)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

REPOSITORY="/content/drive/MyDrive/flan-t5-small-fine-tuned"

# Definimos las opciones del entrenamiento
training_args = Seq2SeqTrainingArguments(
    # Hiperprámetros del entrenamiento
    output_dir=REPOSITORY,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False,  # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=4,
    # Estrategias de logging y evaluación
    logging_dir=f"{REPOSITORY}/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Creamos la instancia de entrenamiento
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=ds_tokens["train"],
    eval_dataset=ds_tokens["validation"],
    compute_metrics=compute_metrics,
)

# Guardamos el tokenizador en disco para utilizarlo posteriormente
tokenizer.save_pretrained(f"{REPOSITORY}/tokenizer")

# Iniciamos el entrenamiento
trainer.train()

### ------------------------ 5. Fine-tuned model text generation and evaluation -------------------------------- ########

'''