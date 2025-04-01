import numpy as np
import evaluate
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from transformers import TrainingArguments
from transformers import Trainer


accuracy = evaluate.load("accuracy")

def compute_metrics_evaluate(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

accuracy = evaluate.load("accuracy")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1}

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def split_dataset(df, label_column):
    """
    Divide un DataFrame en conjuntos de entrenamiento (70%), prueba (20%) y validación (10%).
    
    Parámetros:
    df (pd.DataFrame): El DataFrame de entrada.
    label_column (str): El nombre de la columna de etiqueta para la estratificación.
    
    Retorna:
    DatasetDict: Un diccionario con los conjuntos 'train', 'test' y 'validation'.
    """
    train, test = train_test_split(df, test_size=0.3, stratify=df[label_column])
    test, validation = train_test_split(test, test_size=1/3, stratify=test[label_column])
    
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train, preserve_index=False),
            "test": Dataset.from_pandas(test, preserve_index=False),
            "validation": Dataset.from_pandas(validation, preserve_index=False)
        }
    )
    
    return dataset



def get_training_args(batch_size=32, training_dir="train_dir"):
    return TrainingArguments(
        output_dir=training_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy='epoch'
    )



def create_trainer(model, training_args, compute_metrics, encoded_dataset, tokenizer):
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
        tokenizer=tokenizer
    )
    return trainer



def get_prediction(text):
    input_encoded = tokenizer(text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**input_encoded)

    logits = outputs.logits

    pred = torch.argmax(logits, dim=1).item()
    return id2label[pred]
