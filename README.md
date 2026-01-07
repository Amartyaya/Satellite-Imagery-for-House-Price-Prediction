# Satellite-Imagery-for-House-Price-Prediction

- The data_fetcher.py file downloads the data from SentinelHub. First, the OAuth client ID and client secret need to be obtained from SentinelHub and setup in a .env file as CLIENT_ID and CLIENT_SECRET respectively.

- preprocessing.py handles the preprocessing of the excel files containing the train and test data. preprocessing.ipynb breaks down, and shows step-by-step, the preprocessing stages.

- models.py defines both a dense network utilising the tabular excel data to predict the prices, and a multimodel network containing both a convolutional network and a dense network for predicting the prices.

- training.py contains the training loop for the models.

- model_training.ipynb performs the training of both models and saves the predictions in predictions.xlsx. conv and mlp contain the models obtained after training used to make the predictions.
