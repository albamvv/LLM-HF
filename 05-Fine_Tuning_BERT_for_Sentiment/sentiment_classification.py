import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from pprint import pprint
from transformers import AutoModel 
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import classification_report, confusion_matrix
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline
import numpy as np
import seaborn as sns
from utils import compute_metrics,compute_metrics_evaluate, get_prediction

# Load the CSV file from local storage
df = pd.read_csv("assets/twitter_sentiment.csv")
#df.info() # Display general information about the DataFrame
#print(df.isnull().sum()) # Check for missing values in each column
#print(df['label'].value_counts()) # Count occurrences of each category in the 'label' column

# ----------------- Data analysis ------------------------

# Count the frequency of each category in the 'label_name' column
label_counts = df['label_name'].value_counts(ascending=True)
# Calculate the number of words per tweet
df['Words per Tweet'] = df['text'].str.split().apply(len)
#print("Words per Tweet-> ",df['Words per Tweet'])

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ----------------- Bar Plot: Frequency of Classes ------------------------
label_counts = df['label_name'].value_counts(ascending=True)
label_counts.plot.barh(ax=axes[0])  # Plot on the first subplot
axes[0].set_title("Frequency of Classes")
axes[0].set_xlabel("Count")
axes[0].set_ylabel("Label")

# ----------------- Box Plot: Words per Tweet by Sentiment ------------------------
df.boxplot(column="Words per Tweet", by="label_name", ax=axes[1], grid=False)
axes[1].set_title("Words per Tweet by Sentiment")
axes[1].set_xlabel("Sentiment")
axes[1].set_ylabel("Word Count")

# Adjust layout and remove automatic boxplot title
plt.suptitle("")  
plt.tight_layout()
plt.show() # Show the plots


# ----------------------- Text to Tokens Conversion ----------------------
#- Transformer models like BERT cannot receive raw strings as input; instead, they assume the text has been tokenized and encoded as numerical vectors.
#- Tokenization is the step of breaking down a string into the atomic units used in the model

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "I love machine learning! Tokenization is awesome!!"
encoded_text = tokenizer(text)
#print(encoded_text)
input_ids = tokenizer(text, return_tensors='pt').input_ids  
#print('input_id: ', input_ids)  

#---------- data loader and train test split -----------
train, test = train_test_split(df, test_size=0.3, stratify=df['label_name'])
test, validation = train_test_split(test, test_size=1/3, stratify=test['label_name'])
#train.shape, test.shape, validation.shape

dataset = DatasetDict(
    {'train':Dataset.from_pandas(train, preserve_index=False),
     'test':Dataset.from_pandas(test, preserve_index=False),
     'validation': Dataset.from_pandas(validation, preserve_index=False)
     }
     
)
#print(dataset)
#pprint(dataset['train'][0])
#pprint(dataset['train'][1])

# ------------------------ Tokenization of the Emotion/Sentiment Data

def tokenize(batch):
    temp = tokenizer(batch['text'], padding=True, truncation=True)
    return temp

#print(tokenize(dataset['train'][:2]))
emotion_encoded = dataset.map(tokenize, batched=True, batch_size=None)
pprint(emotion_encoded['train'][0])

# label2id, id2label
label2id = {x['label_name']:x['label'] for x in dataset['train']}
#print(label2id)
id2label = {v:k for k,v in label2id.items()}
#print(id2label)

# ----- Model building -----
model = AutoModel.from_pretrained(model_ckpt)
model.config.id2label

 # --- Fine tunning transformers ---
num_labels = len(label2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config).to(device)



batch_size = 64
training_dir = "bert_base_train_dir"
training_args = TrainingArguments( output_dir=training_dir,
                                  overwrite_output_dir = True,
                                  num_train_epochs = 2,
                                  learning_rate = 2e-5,
                                  per_device_train_batch_size = batch_size,
                                  per_device_eval_batch_size = batch_size,
                                  weight_decay = 0.01,
                                  evaluation_strategy = 'epoch',
                                  disable_tqdm = False
)


#----------- Build model and trainer --------------

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset = emotion_encoded['train'],
                  eval_dataset = emotion_encoded['validation'],
                  tokenizer = tokenizer)

print(trainer.train())

#------------- Model evaluation --------

preds_output = trainer.predict(emotion_encoded['test'])
preds_output.metrics

y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = emotion_encoded['test'][:]['label']


print(classification_report(y_true, y_pred))
print(label2id)


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, xticklabels=label2id.keys(), yticklabels=label2id.keys(), fmt='d', cbar=False, cmap='Reds')
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# ----------- Build prediction function and store model

text = "I am super happy today. I got it done. Finally!!"
get_prediction(text)
trainer.save_model("bert-base-uncased-sentiment-model")

classifier = pipeline('text-classification', model= 'bert-base-uncased-sentiment-model')
classifier([text, 'hello, how are you?', "love you", "i am feeling low"])


