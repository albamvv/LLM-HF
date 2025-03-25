# 1️⃣ Pipeline transformers


## HF Transformers Implementation

This project demonstrates various Natural Language Processing (NLP) and Image Processing tasks using Hugging Face Transformers. It includes text classification, named entity recognition, question answering, text summarization, translation, text generation, and image classification.

[Hugging Face Transformers Quicktour](https://huggingface.co/docs/transformers/v4.35.0/en/quicktour#pipeline)

![alt text](assets/pipeline_summarize.JPG)
---

## Installation
Ensure you have Python installed along with the required libraries:

```sh
pip install transformers pandas pillow requests
```

---

## Implementation Details

### 1. Text Classification
Classifies text sentiment using a pre-trained model.
[SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)

```python
from transformers import pipeline
import pandas as pd

classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
text = "wow! we have come across this far"
outputs = classifier(text)
print("result-> ", pd.DataFrame(outputs))
```
**Expected Output:**
A dataframe with the predicted emotion label and confidence score.
``` sh
      label     score
0  surprise  0.630752
```
---

### 2. Named Entity Recognition (NER)
Extracts key phrases from a given text.
```python
ner_tagger = pipeline("ner", aggregation_strategy="simple", model="ml6team/keyphrase-extraction-kbir-inspec")
text = "Keyphrase extraction is a technique in text analysis where you extract the important keyphrases from a document."
outputs = ner_tagger(text)
print("result-> ", pd.DataFrame(outputs))
```
**Expected Output:**
A dataframe listing extracted keyphrases and their scores.
```sh
  entity_group     score                   word  start  end
0          KEY  0.999997   Keyphrase extraction      0   20
1          KEY  0.999993          text analysis     39   52
```
---

### 3. Question Answering
Extracts answers from a given context.
```python
reader = pipeline("question-answering")
text = "Dear Amazon, last week I ordered an Optimus Prime action figure from your online store in India."
question = "from where did I place order?"
outputs = reader(question=question, context=text)
print("result-> ", pd.DataFrame([outputs]))
```
**Expected Output:**
A dataframe with the extracted answer ("India") and confidence score.
```sh
      score  start  end                 answer
0  0.311888     75   96  online store in India
```
---

### 4. Text Summarization
Summarizes a given text.
```python
summarizer = pipeline("summarization")
outputs = summarizer(text)
print("summarize-> ", outputs)
```
**Expected Output:**
A summarized version of the input text.

---

### 5. Translation (English to German)
Translates English text into German.
```python
translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
outputs = translator(text)
print("translate-> ", outputs)
```
**Expected Output:**
German translation of the input text.

---

### 6. Text Generation
Generates text based on a given prompt.
```python
from transformers import set_seed
set_seed(0)
generator = pipeline("text-generation", model="gpt2-large")
prompt = "There was a lion "
outputs = generator(prompt, max_length=128)
```
**Expected Output:**
A generated story continuing the prompt.
```sh
[{'generated_text': 'There was a lion \xa0(Sudurus rex), a bear ( Ursus americanus ), a cougar\xa0 (Panthera onca ):\nThe second lion was apparently a\xa0 ( Acrocanthosaurus ), but that animal was clearly a subadult male of the\xa0 ( Acrocanthosaurus ) genus and was not clearly a male at all. For some reason people believed that I had photographed a male lion, one of whose most typical characteristics would be one\xa0 which is\xa0 very long and is a bit more heavily muscled. This lion\xa0 (Acrocanthosaurus) was about 40m'}]
```
---

### 7. Image Classification
Classifies an image using a pre-trained model.
```python
from PIL import Image
import requests

url = "https://res.cloudinary.com/dk-find-out/image/upload/q_80,w_1920,f_auto/DCTM_Penguin_UK_DK_AL697473_RGB_PNG_namnse.jpg"
image = Image.open(requests.get(url, stream=True).raw)
classifier = pipeline("image-classification")
outputs = classifier(image)
print(outputs)
```
**Expected Output:**
A list of predicted labels with confidence scores.
```sh
[{'label': 'Egyptian cat', 'score': 0.9214929938316345}, {'label': 'tabby, tabby cat', 'score': 0.058183521032333374}, {'label': 'tiger cat', 'score': 0.012602909468114376}, {'label': 'lynx, catamount', 'score': 0.0037158718332648277}, {'label': 'Siamese cat, Siamese', 'score': 0.00039997967542149127}]
```

### 8.  Image segmentation 
```python
url = "https://img.freepik.com/free-photo/young-bearded-man-with-striped-shirt_273609-5677.jpg"
image = Image.open(requests.get(url, stream=True).raw)
segmenter = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes")
outputs = segmenter(image)
```
**Expected Output:**
A list of predicted labels with confidence scores.
```sh
  score          label                                               mask
0  None     Background  <PIL.Image.Image image mode=L size=626x417 at ...
1  None           Hair  <PIL.Image.Image image mode=L size=626x417 at ...
2  None  Upper-clothes  <PIL.Image.Image image mode=L size=626x417 at ...
3  None           Face  <PIL.Image.Image image mode=L size=626x417 at ...
```

### 9.  Text to speech 

```python
text="""Sam Altman on Wednesday returned to OpenAI as the chief executive officer (CEO) and sacked the Board that had fired him last week. However, the only remaining member in the Board team is Adam D'Angelo, CEO of Quora.
Ex-Salesforce co-CEO Bret Taylor and former US Treasury Secretary and president of Harvard University, Larry Summers will join D'Angelo."""
synth = pipeline("text-to-speech")
speech=synth(text)
sf.write("speech.wav", speech["audio"].T, samplerate=speech['sampling_rate'])
```

### 10. Text to music generation

```python
synth = pipeline("text-to-audio", "facebook/musicgen-small")
text = "a chill song with influences from lofi, chillstep and downtempo"

music = synth(text, forward_params={"do_sample":True})
# sf.write("music.wav",music["audio"].T, samplerate=music['sampling_rate'])
scipy.io.wavfile.write("music.wav", rate=music["sampling_rate"], data=music['audio'])
```

# 2️⃣ LLM Traslate


## Overview
This script loads a pretrained language model to translate text from Chinese to English. It provides different methods for translation: direct model inference and pipeline-based inference.

The script loads a pretrained language model, tokenizes an input text prompt for translation, 
and generates a response using the model. It then decodes and prints the generated output.

[Model: GemmaX2-28-2B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-2B-v0.1)

## Prerequisites
Before running the script, ensure you have the following dependencies installed:
```bash
pip install transformers torch
```

## Implementation Steps

### 1. Load the Pretrained Model and Tokenizer
The script first loads the tokenizer and model using Hugging Face's `transformers` library.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "ModelSpace/GemmaX2-28-2B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

### 2. Tokenize and Generate Translation
The input text (in Chinese) is tokenized and passed through the model to generate translated text.
```python
text = "Translate this from Chinese to English:\nChinese: 我爱机器翻译\nEnglish:"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3. Use a Pipeline for Translation
An alternative method is using the `pipeline` function, which simplifies the translation process.
```python
from transformers import pipeline

pipeline_translator = pipeline("translation", model="ModelSpace/GemmaX2-28-2B-v0.1")
text = "我爱机器翻译"
translated_text = pipeline_translator(text)
print(translated_text)
```

### 4. Alternative Model for Translation
The script also includes an option to use another model (`Helsinki-NLP/opus-mt-zh-en`).
```python
pipeline_translator2 = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
translated_text2 = pipeline_translator2(text)
print(translated_text2[0]['translation_text'])
```

## Running the Script
Execute the script using:
```bash
python 2.LLM_traslate.py
```
This will print translated text for the given Chinese input.



# 3️⃣ LLM Text tokenization


## Overview
This project demonstrates how to tokenize text using the GPT-2 tokenizer and generate predictions using a pretrained GPT-2 language model. The script tokenizes an input sentence, processes it with the model, and extracts probabilities for the next possible tokens.

The script loads the GPT-2 model and tokenizer, tokenizes a given input sentence, and processes it through the model to obtain raw logits,
which represent the probabilities of possible next tokens. Instead of generating full text, it analyzes these logits to determine the most likely next token. 
It also extracts and displays the top ten predicted tokens along with their probabilities, 
making it useful for understanding how the model ranks different token predictions 
and how it assigns likelihoods to various continuations of a given text.

## Prerequisites
Ensure you have the following installed before running the script:
- Python 3.7+
- PyTorch
- Transformers (Hugging Face library)

Install the required dependencies using:
```bash
pip install torch transformers
```

## Implementation Steps

### 1. Import Necessary Libraries
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
```
The script uses the `transformers` library from Hugging Face to handle tokenization and model loading.

### 2. Load the Pretrained GPT-2 Tokenizer and Model
- This loads the GPT-2 tokenizer and the corresponding language model.

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")  
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
```
- This defines the input text that will be tokenized and processed by the model.


```python
sentence = "The future of AI is"
```

### 3. Tokenize the Input
- Tokenize the sentence and convert it into tensor format (PyTorch). 
- Each word or subword is mapped to a specific token ID
- The process: words --> tokens --> unique token IDs --> vector embed

```python
input_ids = tokenizer(sentence, return_tensors='pt').input_ids  
```
This converts the sentence into token IDs, which are then passed as input tensors.

```sh
input_id:  tensor([[ 464, 2003,  286, 9552,  318]])
```
-  Loop through each token ID in the tensor and decode it back to a string

```python
for token_id in input_ids[0]:  
    print(tokenizer.decode(token_id))  # Print the corresponding token  
```

```sh
The    
 future
 of    
 AI    
 is 
```
### 4. Model Prediction
```python
output = gpt2(input_ids)  
final_logits = output.logits[0, -1]
```
The model generates logits, which represent unprocessed probabilities for the next possible tokens.

### 5. Determine the Most Likely Next Token
```python
next_token = tokenizer.decode(final_logits.argmax())
print("Next token prediction:", next_token)
```
This extracts and decodes the most probable next token predicted by the model.

### 6. Generate the Top 10 Predictions
```python
top10 = torch.topk(final_logits.softmax(dim=0), 10)
for value, index in zip(top10.values, top10.indices):
    print(f"{tokenizer.decode(index)} -- {value.item():.1%}")
```
This calculates the probability distribution using softmax and lists the top 10 predictions with their probabilities.

## Running the Script
Save the script as `3.LLM_Text_tokenization.py` and execute it using:
```bash
python 3.LLM_Text_tokenization.py
```
The output will display the next most likely token along with the top 10 predictions and their probabilities.

## Notes
- This script works with the GPT-2 model but can be adapted to other transformer-based models.
- The logits returned are raw values and require softmax transformation to convert them into probabilities.
- Ensure GPU support (if available) by modifying the script to use `.to("cuda")` when loading the model and input tensors.



