from sklearn.model_selection import train_test_split
from pprint import pprint
from transformers import AutoModel 
from imports import*
import seaborn as sns
from utils import compute_metrics,compute_metrics_evaluate, get_prediction,split_dataset, get_training_args, create_trainer

# Load the CSV file from local storage
df = pd.read_csv("assets/twitter_sentiment.csv")
#df.info() # Display general information about the DataFrame
#print(df.isnull().sum()) # Check for missing values in each column
#print(df['label'].value_counts()) # Count occurrences of each category in the 'label' column

# ----------------- Data analysis ------------------------
label_counts = df['label_name'].value_counts(ascending=True) # Count the frequency of each category in the 'label_name' column
df['Words per Tweet'] = df['text'].str.split().apply(len) # Calculate the number of words per tweet
#print("Words per Tweet-> ",df['Words per Tweet'])
# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

'''

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

'''
# ----------------------- Text to Tokens Conversion ----------------------
#- Transformer models like BERT cannot receive raw strings as input; instead, they assume the text has been tokenized and encoded as numerical vectors.
#- Tokenization is the step of breaking down a string into the atomic units used in the model
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
text = "I love machine learning! Tokenization is awesome!!"
encoded_text = tokenizer(text)
input_ids = tokenizer(text, return_tensors='pt').input_ids  

#---------- data loader and train test split -----------
dataset = split_dataset(df, "label_name")  #print(dataset) #pprint(dataset['train'][0]) #pprint(dataset['train'][1])

# ------------------------ Tokenization of the Emotion/Sentiment Data
def tokenize(batch):
    temp = tokenizer(batch['text'], padding=True, truncation=True)
    return temp

#print(tokenize(dataset['train'][:2]))
emotion_encoded = dataset.map(tokenize, batched=True, batch_size=None) 
pprint(emotion_encoded['train'][0])

label2id = {x['label_name']:x['label'] for x in dataset['train']}
id2label = {v:k for k,v in label2id.items()}

# ----- Model building -----
model = AutoModel.from_pretrained(model_ckpt)
#print(model.config.id2label) #print(model.config)

 # --- Fine tunning transformers ---
num_labels = len(label2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config).to(device)
#print(model)

training_dir = "bert_base_train_dir"
training_args = get_training_args(batch_size=64, training_dir=training_dir)

#----------- Build model and trainer --------------
trainer = create_trainer(model, training_args, compute_metrics,emotion_encoded, tokenizer)
#print(trainer.train())

#------------- Model evaluation --------
preds_output = trainer.predict(emotion_encoded['test'])
#print("metrics -> ",preds_output.metrics)
#print("prediction-> ", preds_output.predictions)

y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = emotion_encoded['test'][:]['label']
print("clasification report-> ")
print(classification_report(y_true, y_pred))


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, xticklabels=label2id.keys(), yticklabels=label2id.keys(), fmt='d', cbar=False, cmap='Reds')
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

'''
# ----------- Build prediction function and store model

text = "I am super happy today. I got it done. Finally!!"
get_prediction(text)
trainer.save_model("bert-base-uncased-sentiment-model")

classifier = pipeline('text-classification', model= 'bert-base-uncased-sentiment-model')
classifier([text, 'hello, how are you?', "love you", "i am feeling low"])
'''

