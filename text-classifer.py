from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from config import DATASET_PATH
from processing import clean_text, nltk_preprocess
from model import train_classifier
from utils import load_data, print_results

# code to load data, preprocess, train classifier, evaluate, etc.
