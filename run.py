
import re
import numpy as np
import pandas as pd
import json
import torch
import torch.optim as optim
import torch.nn as nn

from resources.Vectorizer import ReviewVectorizer
from resources.Dataset import ReviewDataset, generate_batches
from resources.Model import ReviewClassifier

from train_val_test_model import train_val_model
from config import config

def set_seeds(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def run_training():
    if not torch.cuda.is_available():
        cuda = False
    device = torch.device("cuda" if cuda else "cpu")

    set_seeds(config.seed, cuda)

    # dataset and vectorizer
    dataset = ReviewDataset(config.data_path)
    vectorizer = dataset.get_vectorizer()

    # model
    classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
    classifier = classifier.to(device)

    # loss and optimizer
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=config.learning_rate)

    train_state, classifier, optimizer = train_val_model(classifier, dataset, device,  optimizer, loss_func)

    print(train_state)

if __name__ == "__main__":
    run_training()

