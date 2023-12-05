import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import regexpTokenizer, word_tokenize
import nltk

#  necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
sw=stopwords.words('english')
tokenizer=regexpTokenizer(r'\w+')

# Function to clean text data
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', 'number', text)  # Replace numbers
    return text

# Function to preprocess data using NLTK
def nltk_preprocess(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Removing stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
