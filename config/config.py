
problem = 'IMDB_review_sentiment'

if problem == 'IMDB_review_sentiment':
    frequency_cutoff=25
    model_filename = 'single_layer_perceptron'
    data_path='data/IMDB_Dataset.csv'
    save_dir='model/'

    text_column = 'review'
    y_column = 'sentiment'

    # Training hyperparameters
    batch_size=300
    early_stopping_criteria=5
    learning_rate=0.001
    num_epochs=1
    seed=1337





