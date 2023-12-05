# Configuration settings
DATASET_PATH = '../Youtube03-LMFAO.csv'
rs = 42  # random state
stem ="porter" # porter, snowball, or lancaster
min_df = 2  # minimum document frequency
ng_low = 1  # lower bound of n-gram range
ng_high = 2  # upper bound of n-gram range
max_df = 0.95  # maximum document frequency