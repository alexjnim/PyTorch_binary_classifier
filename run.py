
import re
import numpy as np
import pandas as pd
import json
import torch
import torch.optim as optim
import torch.nn as nn

from resources.Vectorizer import TextVectorizer
from resources.Dataset import TextDataset, generate_batches
from model.SingleLayerPerceptron import SingleLayerPerceptron
from model.MultiLayerPerceptron import MultiLayerPerceptron

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
    dataset = TextDataset(config.data_path)
    vectorizer = dataset.get_vectorizer()

    if config.model == 'single_layer_perceptron':
        # model
        classifier = SingleLayerPerceptron(num_features=len(vectorizer.text_vocab))
        classifier = classifier.to(device)

        # loss and optimizer
        loss_func = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=config.learning_rate)
    elif config.model == 'multi_layer_perceptron':
        # model
        classifier = MultiLayerPerceptron(input_dim=len(vectorizer.text_vocab), hidden_dim=50, output_dim=1)
        classifier = classifier.to(device)

        # loss and optimizer
        loss_func = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=config.learning_rate)

    train_state, classifier, optimizer = train_val_model(classifier, dataset, device,  optimizer, loss_func)
    print(train_state)

if __name__ == "__main__":
    run_training()

