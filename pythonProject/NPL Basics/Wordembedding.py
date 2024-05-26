from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import re
import pandas as pd
import numpy as np


df = pd.read_csv('emails.csv')
clean_txt = []
for w in range(len(df.text)):
   desc = df['text'][w].lower()

   #remove punctuation
   desc = re.sub('[^a-zA-Z]', ' ', desc)

   #remove tags
   desc=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)

   #remove digits and special chars
   desc=re.sub("(\\d|\\W)+"," ",desc)
   clean_txt.append(desc)

df['clean'] = clean_txt

# Tokenize the cleaned text
sentences = [text.split() for text in df["clean"].tolist()]

# Train a Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)


# Extract the word vectors from the model
words = list(model.wv.index_to_key)
word_vectors = np.array([model.wv[word] for word in words])

# Reduce dimensions to 2D using PCA
pca = PCA(n_components=2)
result = pca.fit_transform(word_vectors)

# Create a scatter plot of the projections
import plotly.express as px
fig = px.scatter(x=result[:, 0], y=result[:, 1], text=words)
fig.update_traces(textposition="top center")
fig.show()