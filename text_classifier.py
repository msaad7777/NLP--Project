import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import TfidfVectorizer
#  necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load and explore data
data = pd.read_csv('./Youtube03-LMFAO.csv')
print(data.head())
print(data.info())
# Load stopwords into a set for faster access
stop_words = set(stopwords.words('english'))
#data.dropna(inplace=True)
# Function to clean text data
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', 'number', text)  # Replace numbers
    return text

# Function to preprocess data using NLTK
def nltk_preprocess(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Removing stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Apply cleaning and preprocessing to the CONTENT column
data['CONTENT'] = data['CONTENT'].apply(clean_text)
data['CONTENT'] = data['CONTENT'].apply(nltk_preprocess)

# Extracting features from text
count_vectorizer = CountVectorizer()
# count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2))
# count_vectorizer = CountVectorizer(ngram_range=(1, 2))  # for bi-grams
# count_vectorizer = CountVectorizer(min_df=200) 

# tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
# X_tfidf = tfidf_vectorizer.fit_transform(data['CONTENT'])

X_counts = count_vectorizer.fit_transform(data['CONTENT'])
print("Shape of Count Vectors:", X_counts.shape)


feature_names = count_vectorizer.get_feature_names_out()

# Print a sample of feature names (tokens)
print("Sample feature names:", feature_names[:10])  

# Summarize the occurrence of the tokens
token_counts = X_counts.sum(axis=0)
token_frequency = [(feature_names[i], token_counts[0, i]) for i in range(token_counts.shape[1])]
sorted_token_frequency = sorted(token_frequency, key=lambda x: x[1], reverse=True)

# Print the 10 most common tokens
print("Most common tokens:", sorted_token_frequency[:10])

# Applying TF-IDF
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
print("Shape of TFIDF Vectors:", X_tfidf.shape)

# Display some sample TF-IDF values for the first document
print("Sample of TF-IDF values for the first document:")
print(X_tfidf[0])

# Display the average TF-IDF value for words in all documents (optional)
average_tfidf = X_tfidf.mean(axis=0)
print("Average TF-IDF value for all words in the corpus:")
print(average_tfidf)


# Shuffle and split data
data_shuffled = data.sample(frac=1).reset_index(drop=True)
X = X_tfidf
y = data_shuffled['CLASS']
train_size = int(0.75 * len(data))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# # Model training
# classifier = MultinomialNB().fit(X_train, y_train)
# Train classifier with hyperparameter tuning
classifier = MultinomialNB(alpha=0.1).fit(X_train, y_train)

# Cross-validation
# accuracy_scores = cross_val_score(classifier, X_train, y_train, cv=5)
# print("Mean accuracy from cross-validation:", accuracy_scores.mean())

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
accuracy_scores = cross_val_score(classifier, X_train, y_train, cv=skf)
print("Mean accuracy from cross-validation:", accuracy_scores.mean())

# Model testing
y_pred = classifier.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# Test the model with new comments
new_comments = [
    'Check this website, www.canadaforme.ca',  # non-spam
    'This video is very well made, kudos to the director.',  # non-spam
    'Check out my profile for free music!',  # spam
    'Amazing content as always, just subscribed!',  # non-spam
    'Great tutorial, learned a lot from this.',  # non-spam
    'Free games and gift cards, check my profile!',  # spam
]

# Preprocess new comments
new_comments_processed = [clean_text(comment) for comment in new_comments]
new_comments_processed = [nltk_preprocess(comment) for comment in new_comments_processed]

# Transform new comments using the same vectorizer and transformer
new_comments_counts = count_vectorizer.transform(new_comments_processed)
new_comments_tfidf = tfidf_transformer.transform(new_comments_counts)

# Predict the class for new comments
new_predictions = classifier.predict(new_comments_tfidf)
predictions_labels = ['Spam' if label == 1 else 'Not Spam' for label in new_predictions]

# Print the predictions
for comment, prediction in zip(new_comments, predictions_labels):
    print('\nComment:', comment, '\nPredicted category:', prediction)






