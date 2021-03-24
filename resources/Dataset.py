from collections import Counter
import json
import os
import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from resources.Vectorizer import TextVectorizer
from resources.NormalizeText import normalize_corpus, simple_normalize
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, data_path):
        """
        Args:
            text_df (pandas.DataFrame): the dataset
        """
        self.text_df = pd.read_csv(data_path)

        #simple normalizer
        self.text_df.review = self.text_df.review.apply(simple_normalize)
        # this normalizing function takes too long
        # self.text_df.review = normalize_corpus(corpus=self.text_df.review, html_stripping=True,
        #                                  contraction_expansion=True, accented_char_removal=True,
        #                                  text_lower_case=True, text_lemmatization=True,
        #                                  text_stemming=False, special_char_removal=True,
        #                                  remove_digits=True, stopword_removal=True)

        train_df, self.test_df = train_test_split(self.text_df, test_size=0.1, random_state=42, shuffle=True)
        self.train_df, self.val_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True)

        self.train_size = len(self.train_df)
        self.validation_size = len(self.val_df)
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')
        #here we used the imported vectorizer
        self._vectorizer = TextVectorizer.from_dataframe(self.train_df)


    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe

        Args:
            split (str): one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        review_vector = \
            self._vectorizer.vectorize(row.review)

        rating_index = \
            self._vectorizer.rating_vocab.lookup_token(row.sentiment)

        return {'x_data': review_vector,
                'y_target': rating_index}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict