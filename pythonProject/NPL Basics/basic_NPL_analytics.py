import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")

#Setting-Up the tasks
# Load the CSV file
df = pd.read_csv("Therapy_bot_Replies.csv")

# Get the correct column with the text corpus for each response
texts = df["response_text"].tolist()

# first Transformation function
def custom_preprocessor(text):
    # Lowercase and remove accents for the vectorizer
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove URLs
    text = re.sub(r"http\S+", "url", text)
    return text

# second Transformation function for lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_tokenize(text):
    return [lemmatizer.lemmatize(word) for word in text.split()]



# Initialize the CountVectorizer and transform the corpus
vectorizer = CountVectorizer()
lowercase = True,
strip_accents = "unicode",
preprocessor = custom_preprocessor,




# Initialize the CountVectorizer with the subliner term frequency and transform the corpus
second_vectorizer = TfidfVectorizer(sublinear_tf=True)
lowercase = True,
strip_accents = "unicode",
preprocessor = custom_preprocessor,




#Task 1: Bag of Words Count
# Fit and transform the texts
X = vectorizer.fit_transform(texts)

# Convert the result to a DataFrame
word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Filter out columns where the sum of counts is zero (i.e., no occurrences of the word)
word_counts = word_counts.loc[:, (word_counts.sum() > 0)]

# Visualisazion of the Count of Words
sum_words = word_counts.sum(axis=0)
sum_words.sort_values(ascending=False).head(20).plot(kind="bar", title="Top 20 words")
plt.xlabel("Words")
plt.ylabel("Count")
plt.show()



# Task 2: Binary Search Function
def search_word(word):
    # Check if the word is in the vectorizer's vocabulary
    if word in vectorizer.vocabulary_:
        # Get the index of the word
        word_index = vectorizer.vocabulary_[word]
        # Calculate the total count of the word across all responses
        word_count = word_counts.iloc[:, word_index].sum()
        # Return True (found) and the count
        return True, word_count
    else:
        # Return False (not found) and count as 0
        return False, 0



found, count = search_word("friend")
print("Found:", found)
print("Count:", count)



#task 3
# Fit and transform the texts using the TF-IDF vectorizer
Y = second_vectorizer.fit_transform(texts)

# Convert the result to a DataFrame
tfidf_counts = pd.DataFrame(Y.toarray(), columns=second_vectorizer.get_feature_names_out())

# Filter out columns where the sum of counts is zero
tfidf_counts = tfidf_counts.loc[:, (tfidf_counts.sum() > 0)]

# Visualization of the TF-IDF of Words
sum_tfidf = tfidf_counts.sum(axis=0)
sum_tfidf.sort_values(ascending=False).head(20).plot(kind="bar", title="Top 20 words by TF-IDF")
plt.xlabel("Words")
plt.ylabel("TF-IDF")
plt.show()


# Task 4 Checking Frequency of Words in each Document
# Variation 1: Term Frequency (TF) Only
tf_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Variation 2: Inverse Document Frequency (IDF)
# Calculate IDF
total_documents = len(texts)
idf_values = np.log(total_documents / (word_counts > 0).sum(axis=0))
idf_matrix = tf_matrix.multiply(idf_values, axis=1)

# Variation 3: Smooth Inverse Document Frequency (Smooth IDF)
# Calculate Smooth IDF
smooth_idf_values = np.log((1 + total_documents) / (1 + (word_counts > 0).sum(axis=0))) + 1
smooth_idf_matrix = tf_matrix.multiply(smooth_idf_values, axis=1)

# Display the matrices
plt.figure(figsize=(10, 8))
sns.heatmap(tf_matrix.iloc[:20, :20], annot=True, fmt="d", cmap="YlGnBu")
plt.title("Term Frequency Matrix")
plt.xlabel("Terms")
plt.ylabel("Documents")
plt.show()

idf_values.plot(kind="bar", figsize=(10, 6))
plt.title("Inverse Document Frequency (IDF) Values")
plt.xlabel("Terms")
plt.ylabel("IDF Score")
plt.show()