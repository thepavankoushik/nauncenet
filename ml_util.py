import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import ml


tweets = pd.read_csv('data/train.csv', sep=',')
selected_columns = ['text', 'target']
twitter_df = tweets[selected_columns]


tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(twitter_df['text'])

# Load the models
logistic_regression_model = pickle.load(open('models/logistic_regression_model.sav', 'rb'))
naive_bayes_model = pickle.load(open('models/naive_bayes_model.sav', 'rb'))
xgboost_model = pickle.load(open('models/xgboost_model.sav', 'rb'))
svm_model = pickle.load(open('models/svm_model.sav', 'rb'))
random_forest_model = pickle.load(open('models/random_forest_model.sav', 'rb'))


def ml_classify_tweet(tweet, model, vectorizer=tfidf_vectorizer):
    tweet_vector = vectorizer.transform([tweet])
    prediction = model.predict(tweet_vector)
    if prediction[0] == 1:
      return 'Disaster'
    else:
      return 'Not Disaster'


if __name__ == "__main__":
   print(ml_classify_tweet("I love my life", logistic_regression_model))