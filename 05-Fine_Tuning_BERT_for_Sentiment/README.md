# Sentiment Classification using Transformers

This project performs sentiment classification using transformer models, specifically leveraging BERT (`bert-base-uncased`). The dataset consists of tweets labeled with sentiment categories. 

## Installation

Ensure you have the necessary dependencies installed:

```bash
pip install pandas matplotlib transformers datasets scikit-learn torch
```

## Steps to Run the Code

### 1. Load and Analyze Data

Load the dataset:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("assets/twitter_sentiment.csv")
```

Perform basic analysis:

```python
print(df.info())  # General dataset information
print(df.isnull().sum())  # Check for missing values
print(df['label'].value_counts())  # Count occurrences of each label
```

### 2. Visualizing Data

Plot class distribution and word count per tweet:

```python
label_counts = df['label_name'].value_counts(ascending=True)
df['Words per Tweet'] = df['text'].str.split().apply(len)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
label_counts.plot.barh(ax=axes[0])
axes[0].set_title("Frequency of Classes")
axes[0].set_xlabel("Count")
axes[0].set_ylabel("Label")

df.boxplot(column="Words per Tweet", by="label_name", ax=axes[1], grid=False)
axes[1].set_title("Words per Tweet by Sentiment")
axes[1].set_xlabel("Sentiment")
axes[1].set_ylabel("Word Count")

plt.suptitle("")  
plt.tight_layout()
plt.show()
```

### 3. Tokenization

Use BERT tokenizer to convert text into numerical representation:

```python
from transformers import AutoTokenizer

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

encoded_text = tokenizer("I love machine learning! Tokenization is awesome!!")
print(encoded_text)
```

### 4. Data Splitting and Conversion to Dataset

Split the dataset into training, testing, and validation sets:

```python
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

train, test = train_test_split(df, test_size=0.3, stratify=df['label_name'])
test, validation = train_test_split(test, test_size=1/3, stratify=test['label_name'])

dataset = DatasetDict({
    'train': Dataset.from_pandas(train, preserve_index=False),
    'test': Dataset.from_pandas(test, preserve_index=False),
    'validation': Dataset.from_pandas(validation, preserve_index=False)
})
```

Apply tokenization to the dataset:

```python
dataset = dataset.map(tokenize, batched=True, batch_size=None)
```

### 5. Mapping Labels

Create mappings between labels and numerical IDs:

```python
label2id = {x['label_name']: x['label'] for x in dataset['train']}
id2label = {v: k for k, v in label2id.items()}
print(label2id)
print(id2label)
```

### 6. Load Pretrained Model

Load the BERT model for sequence classification:

```python
from transformers import AutoModelForSequenceClassification, AutoConfig
import torch

num_labels = len(label2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config).to(device)
```

## Conclusion

This project sets up a sentiment classification pipeline using BERT, covering data loading, preprocessing, tokenization, and model initialization. The next steps involve training and evaluation.
