import torch
import re
from model.SingleLayerPerceptron import SingleLayerPerceptron
from resources.Vectorizer import TextVectorizer
from resources.Dataset import TextDataset
from resources.NormalizeText import simple_normalize
from config import config


def predict_rating(review, model, vectorizer, decision_threshold=0.5):
    """Predict the rating of a review

    Args:
        review (str): the text of the review
        model (ReviewClassifier): the trained model
        vectorizer (ReviewVectorizer): the corresponding vectorizer
        decision_threshold (float): The numerical boundary which separates the rating classes
    """
    review = simple_normalize(review)
    # can use more complicated normalize_corpus fucntion here from NormalizeText.py

    vectorized_review = torch.tensor(vectorizer.vectorize(review))
    result = model(vectorized_review.view(1, -1))

    probability_value = torch.sigmoid(result).item()
    index = 1
    if probability_value < decision_threshold:
        index = 0

    return probability_value, vectorizer.rating_vocab.lookup_index(index)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        cuda = False
    device = torch.device("cuda" if cuda else "cpu")

    dataset = TextDataset(config.data_path)
    vectorizer = dataset.get_vectorizer()

    model = SingleLayerPerceptron(num_features=len(vectorizer.review_vocab))
    model.load_state_dict(torch.load(config.save_dir+config.model_filename))
    model = model.to(device)

    review = 'I think it was pretty good'

    probability_value, sentiment = predict_rating(review, model, vectorizer, decision_threshold=0.5)
    print(probability_value)
    print(sentiment)
