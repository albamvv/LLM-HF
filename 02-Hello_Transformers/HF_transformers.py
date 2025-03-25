from transformers import pipeline  # Import the pipeline function from transformers
import pandas as pd  # Import pandas for handling tabular data
from PIL import Image  # Import PIL for image processing
import requests  # Import requests to fetch images from URLs
from transformers import set_seed  # Import set_seed for reproducibility

##----------------------- Text classification ----------------
# Load a text classification pipeline with a specified model
model="SamLowe/roberta-base-go_emotions"
classifier = pipeline("text-classification", model=model)
text = "wow! we have come across this far"
outputs = classifier(text)  # Classify the input text
#print(pd.DataFrame(outputs))# Print results as a DataFrame

##----------------------- Named Entity Recognition (NER) ----------------
# Load an NER pipeline with an aggregation strategy
#ner_tagger = pipeline("ner", aggregation_strategy="simple")
ner_tagger = pipeline("ner", aggregation_strategy="simple", model="ml6team/keyphrase-extraction-kbir-inspec")
text = "Keyphrase extraction is a technique in text analysis where you extract the important keyphrases from a document. Thanks to these keyphrases humans can understand the content of a text very quickly and easily without reading it completely."
outputs = ner_tagger(text)  # Perform NER on the input text
#print((pd.DataFrame(outputs))) # Print results



##----------------------- Question Answering -------------------------
text = """
Dear Amazon, last week I ordered an Optimus Prime action figure from your
online store in India. Unfortunately when I opened the package, I discovered to
my horror that I had been sent an action figure of Megatron instead!
"""
reader = pipeline("question-answering")  # Load a question-answering pipeline
question = "from where did I place the order?"
outputs = reader(question=question, context=text)  # Answer the question based on the provided context
#print(pd.DataFrame([outputs]))  # Print results


##----------------------- Summarization -------------------------
summarizer = pipeline("summarization")  # Load a summarization pipeline
outputs = summarizer(text)  # Summarize the given text
#print("summarize-> ", outputs)  # Print the summary


##----------------------- Translation -------------------------
translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")  # Load an English-to-German translation pipeline
outputs = translator(text)  # Translate the text
#print("translate-> ", outputs)  # Print the translated text


##----------------------- Text Generation -------------------------
set_seed(0)  # Set seed for reproducibility
generator = pipeline("text-generation", model="gpt2-large")  # Load a text-generation model
response = "I am sorry to hear that your order was mixed up"
#prompt= "user: " + text +" Customer Service Response. " + response
prompt = "There was a lion "  # Provide a text prompt for generation
outputs = generator(prompt, max_length=128)  # Generate text based on the prompt
print("generated text-> ", outputs)  # Print the generated text


##----------------------- Image classification -------------------------
# Load an image classification pipeline and classify two images from URLs
classifier = pipeline("image-classification")
url = "https://res.cloudinary.com/dk-find-out/image/upload/q_80,w_1920,f_auto/DCTM_Penguin_UK_DK_AL697473_RGB_PNG_namnse.jpg"
image = Image.open(requests.get(url, stream=True).raw)  # Fetch and open the image
outputs = classifier(image)  # Classify the image
print("image classification-> ", outputs)  # Print the classification results

url = "https://img.freepik.com/free-photo/young-bearded-man-with-striped-shirt_273609-5677.jpg"
image = Image.open(requests.get(url, stream=True).raw)  # Fetch and open another image
classifier = pipeline("image-classification", model="nateraw/vit-age-classifier")  # Load a different classification model
outputs = classifier(image)  # Classify the image
print("age classification-> ", outputs)  # Print the classification results
