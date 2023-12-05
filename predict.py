import os
import config
import argparse
from source.utils import load_file
from source.processing import process_text
from sklearn.linear_model import LogisticRegression

def main(args):
    