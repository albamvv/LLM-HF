from transformers import pipeline
import pandas as pd
from PIL import Image
import requests
from transformers import pipeline
from transformers import set_seed



##----------------------- Text classification ----------------
classifier = pipeline("text-classification", model = "SamLowe/roberta-base-go_emotions")
text = "wow! we have come across this far"
outputs = classifier(text)
print("result-> ",pd.DataFrame(outputs))

##----------------------- Name Entity recognition ----------------

ner_tagger = pipeline("ner", aggregation_strategy="simple", model="ml6team/keyphrase-extraction-kbir-inspec")
text = "Keyphrase extraction is a technique in text analysis where you extract the important keyphrases from a document.  Thanks to these keyphrases humans can understand the content of a text very quickly and easily without reading  it completely. "
outputs = ner_tagger(text)
print("result-> ",pd.DataFrame(outputs))

##----------------------- Question answering -------------------------

text = """
Dear Amazon, last week I ordered an Optimus Prime action figure from your
online store in India. Unfortunately when I opened the package, I discovered to
my horror that I had been sent an action figure of Megatron instead!
"""
reader = pipeline("question-answering")
question = "from where did I placed order?"
outputs = reader(question=question, context=text)
print("result-> ",pd.DataFrame([outputs]))


##----------------------- Summarization -------------------------
summarizer = pipeline("summarization")
outputs = summarizer(text)
print("summarize-> ",outputs)


##----------------------- Traslation -------------------------
translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
outputs = translator(text)
print("traslate-> ",outputs)

##----------------------- Text Generation -------------------------
set_seed(0)
generator = pipeline("text-generation", model="gpt2-large")
response = "I am sorry to hear that your order was mixed up"
# prompt = "user: " + text.replace("\n", " ") + " Customer Service Response: " + response

prompt = "There was a lion "
outputs = generator(prompt, max_length=128)

##----------------------- Image classification -------------------------
url = "https://res.cloudinary.com/dk-find-out/image/upload/q_80,w_1920,f_auto/DCTM_Penguin_UK_DK_AL697473_RGB_PNG_namnse.jpg"
image = Image.open(requests.get(url, stream=True).raw)
classifier = pipeline("image-classification")
outputs = classifier(image)
outputs

url = "https://img.freepik.com/free-photo/young-bearded-man-with-striped-shirt_273609-5677.jpg"
image = Image.open(requests.get(url, stream=True).raw)
classifier = pipeline("image-classification", model="nateraw/vit-age-classifier")
outputs = classifier(image)
outputs
