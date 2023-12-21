from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


tweets = pd.read_csv('data/train.csv', sep=',')
selected_columns = ['text', 'target']
twitter_df = tweets[selected_columns]

# Initialize the vectorizer
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(twitter_df['text'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, twitter_df['target'], test_size=0.2, random_state=42)

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)

xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgboost_model.fit(X_train, y_train)

svm_model = SVC()
svm_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)

# Save the models
import pickle
pickle.dump(logistic_regression_model, open('models/logistic_regression_model.sav', 'wb'))
pickle.dump(naive_bayes_model, open('models/naive_bayes_model.sav', 'wb'))
pickle.dump(xgboost_model, open('models/xgboost_model.sav', 'wb'))
pickle.dump(svm_model, open('models/svm_model.sav', 'wb'))
pickle.dump(random_forest_model, open('models/random_forest_model.sav', 'wb'))
