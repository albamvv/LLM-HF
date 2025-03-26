import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file from local storage
df = pd.read_csv("assets/twitter_sentiment.csv")
#df.info() # Display general information about the DataFrame
#print(df.isnull().sum()) # Check for missing values in each column
#print(df['label'].value_counts()) # Count occurrences of each category in the 'label' column

# ----------------- Data analysis ------------------------

# Count the frequency of each category in the 'label_name' column
label_counts = df['label_name'].value_counts(ascending=True)

# Plot the distribution of classes
label_counts.plot.barh()
plt.title("Frequency of Classes")
plt.xlabel("Count")
plt.ylabel("Label")
plt.show()

print(df['Words per Tweet'] = df['text'].str.split().apply(len))
print(df.boxplot("Words per Tweet", by="label_name"))


# ----------------------- Text to Tokens Conversion ----------------------
#- Transformer models like BERT cannot receive raw strings as input; instead, they assume the text has been tokenized and encoded as numerical vectors.
#- Tokenization is the step of breaking down a string into the atomic units used in the model