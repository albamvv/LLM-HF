# Fine Tuning DistilBERT, MobileBERT and TinyBERT for Fake News Detection
from utils import split_dataset, compute_metrics_evaluate, get_training_args, create_trainer
from imports import*

# Load the CSV file from local storage
df = pd.read_csv("assets/fake_news.csv")

# ----------------- Data analysis ------------------------
label_counts = df['label'].value_counts(ascending=True)
label_counts.plot.barh()
plt.title("Frequency of Classes")
plt.show()

# 1.5 tokens per word on average
df['title_tokens'] = df['title'].apply(lambda x: len(x.split())*1.5)
df['text_tokens'] = df['text'].apply(lambda x: len(x.split())*1.5)


fig, ax = plt.subplots(1,2, figsize=(15,5))

ax[0].hist(df['title_tokens'], bins=50, color = 'skyblue')
ax[0].set_title("Title Tokens")

ax[1].hist(df['text_tokens'], bins=50, color = 'orange')
ax[1].set_title("Text Tokens")

plt.show()


#---------- data loader and train test split -----------
dataset = split_dataset(df, "label")

#---------- Data tokenization -----------
text = "Machine learning is awesome!! Thanks KGP Talkie."

model_ckpt = "distilbert-base-uncased"
distilbert_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
distilbert_tokens = distilbert_tokenizer.tokenize(text)

model_ckpt = "google/mobilebert-uncased"
mobilebert_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
mobilebert_tokens = mobilebert_tokenizer.tokenize(text)

model_ckpt = "huawei-noah/TinyBERT_General_4L_312D"
tinybert_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tinybert_tokens = tinybert_tokenizer.tokenize(text)


def tokenize(batch):
    temp = distilbert_tokenizer(batch['title'], padding=True, truncation=True)
    return temp

print(tokenize(dataset['train'][:2]))
encoded_dataset = dataset.map(tokenize, batch_size=None, batched=True)

##--------------- Model building-------------
label2id = {"Real": 0, "Fake": 1}
id2label = {0:"Real", 1:"Fake"}

model_ckpt = "distilbert-base-uncased"
# model_ckpt = "google/mobilebert-uncased"
# model_ckpt = "huawei-noah/TinyBERT_General_4L_312D"

num_labels = len(label2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config).to(device)

##--------------- Model Fine tuning-------------
batch_size = 32
training_dir = "train_dir"
training_args = get_training_args(batch_size=32, training_dir=training_dir)
trainer = create_trainer(model, training_args, compute_metrics_evaluate, encoded_dataset, distilbert_tokenizer)

##-------------- Model evaluation ---------------
preds_output = trainer.predict(encoded_dataset['test'])
y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = encoded_dataset['test'][:]['label']
print(classification_report(y_true, y_pred, target_names=list(label2id)))



