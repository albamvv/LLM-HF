# Sentiment Classification using Transformers

This project performs sentiment classification using transformer models, specifically leveraging BERT (`bert-base-uncased`). The dataset consists of tweets labeled with sentiment categories, and we will walk through each step from loading data to preparing the model.

## Installation

Before running the script, ensure you have the required Python libraries installed. You can do this by running:

```bash
pip install pandas matplotlib transformers datasets scikit-learn torch
```

## Steps to Run the Code

### 1. Load and Analyze Data

The dataset used in this project is stored as a CSV file. We first load it into a Pandas DataFrame:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("assets/twitter_sentiment.csv")
```

#### Data Exploration

To understand the structure of the dataset, we can check its information:

```python
print(df.info())  # Displays the number of rows, columns, and data types
print(df.isnull().sum())  # Checks for missing values in each column
print(df['label'].value_counts())  # Shows the count of each sentiment category
```

### 2. Visualizing Data

To gain insights into the data distribution, we create plots for class frequency and word count per tweet.

#### Plot Class Distribution

```python
label_counts = df['label_name'].value_counts(ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
label_counts.plot.barh(ax=axes[0])  # Horizontal bar plot
axes[0].set_title("Frequency of Classes")
axes[0].set_xlabel("Count")
axes[0].set_ylabel("Label")
```

#### Plot Word Count Distribution

We calculate the number of words per tweet and create a box plot:

```python
df['Words per Tweet'] = df['text'].str.split().apply(len)
df.boxplot(column="Words per Tweet", by="label_name", ax=axes[1], grid=False)
axes[1].set_title("Words per Tweet by Sentiment")
axes[1].set_xlabel("Sentiment")
axes[1].set_ylabel("Word Count")

plt.suptitle("")  # Remove automatic title
plt.tight_layout()
plt.show()
```
![Alt text](assets/data_visualization.JPG)

### 3. Tokenization

Transformer models like BERT cannot process raw text directly. Instead, text must be tokenized and converted into numerical vectors. We use the BERT tokenizer for this:

```python
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
```

To see how tokenization works, let’s tokenize an example sentence:

```python
text = "I love machine learning! Tokenization is awesome!!"
encoded_text = tokenizer(text)
print(encoded_text)  # Tokenized representation of text
```
**Ouput**
```sh 
{'input_ids': [101, 1045, 2293, 3698, 4083, 999, 19204, 3989, 2003, 12476, 999, 999, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

```python
input_ids = tokenizer(text, return_tensors='pt').input_ids  
print('input_id: ', input_ids)  
```

**Ouput**
```sh 
input_id:  tensor([[  101,  1045,  2293,  3698,  4083,   999, 19204,  3989,  2003, 12476,
           999,   999,   102]])
```
### 4. Splitting Data and Creating Dataset

To train the model, we split the dataset into training, validation, and testing sets:

```python
train, test = train_test_split(df, test_size=0.3, stratify=df['label_name'])
test, validation = train_test_split(test, test_size=1/3, stratify=test['label_name'])
```

We then convert these into the Hugging Face `DatasetDict` format:

```python
dataset = DatasetDict({
    'train': Dataset.from_pandas(train, preserve_index=False),
    'test': Dataset.from_pandas(test, preserve_index=False),
    'validation': Dataset.from_pandas(validation, preserve_index=False)
})
```

**Ouput:**
```sh
DatasetDict({
    train: Dataset({
        features: ['text', 'label', 'label_name', 'Words per Tweet'],
        num_rows: 11200
    })
    test: Dataset({
        features: ['text', 'label', 'label_name', 'Words per Tweet'],
        num_rows: 3200
    })
    validation: Dataset({
        features: ['text', 'label', 'label_name', 'Words per Tweet'],
        num_rows: 1600
    })
})
```
```python
pprint(dataset['train'][0])
```
**Output**
```sh
{'Words per Tweet': 9,
 'label': 1,
 'label_name': 'joy',
 'text': 'i think we ll feel pretty good about that'}
```
```python
pprint(dataset['train'][1])
```

**Output**
```sh
{'Words per Tweet': 32,
 'label': 3,
 'label_name': 'anger',
 'text': 'i feel like i m so distracted by silly things like twitter that i '
         'can spend an entire evening with the kids and not actually hear a '
         'thing that they re saying'}
 'text': 'i feel like i m so distracted by silly things like twitter that i '
         'can spend an entire evening with the kids and not actually hear a '
         'thing that they re saying'}
```

Next, we apply tokenization to the dataset:

```python
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True, batch_size=None)
```

### 5. Mapping Labels

Since machine learning models work with numerical labels, we create mappings:

**Dictionary mapping label names to numerical IDs**
- The resulting dictionary maps sentiment names to numerical IDs. 
```python
label2id = {x['label_name']: x['label'] for x in dataset['train']}
print(label2id)   
```
Output:

```sh
{'sadness': 0, 'joy': 1, 'surprise': 5, 'anger': 3, 'love': 2, 'fear': 4}
```
**Reverse mapping from IDs to labels**
- The new dictionary maps numerical IDs back to sentiment labels.

```python
id2label = {v: k for k, v in label2id.items()}
print(id2label)  
```
Ouput: 
```sh
{0: 'sadness', 1: 'joy', 5: 'surprise', 3: 'anger', 2: 'love', 4: 'fear'}
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
```
### 6. Load Pretrained Model

We now load the BERT model for sequence classification and configure it to recognize our label mappings:

```python
num_labels = len(label2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config).to(device)
```

