from ml_util import *
from llm_util import *
import gradio as gr
import time

def predict(model, tweet):
    classic_ml = [logistic_regression_model, naive_bayes_model, xgboost_model, svm_model, random_forest_model]
    tf = [llama2_70b, falcon_7b, mistral_7b]
    classic_dl = []
    if model in classic_ml:
      return ml_classify_tweet(tweet, model, tfidf_vectorizer)
    elif model in tf:
        return llm_classify_tweet(model, tweet)
    return

def classify_tweet(tweet, expected):
    results = {}
    models = {
        'Logistic Regression': logistic_regression_model,
        'Naive Bayes': naive_bayes_model,
        'XGBoost': xgboost_model,
        'SVM': svm_model,
        'Random Forest': random_forest_model,
        # Transformer Models
        'LLaMa2_70B': llama2_70b,
        'Falcon_7B': falcon_7b,
        'Mistral_7B': mistral_7b
    }

    for model_name, model in models.items():
        prediction = predict(model, tweet)
        time.sleep(2)
        match = 'green' if prediction == expected else 'red'
        results[model_name] = f"<div style='color: {match}; border:2px solid {match}; padding:5px; margin:2px;'>{model_name}: {prediction}/{expected}</div>"

    return list(results.values())

iface = gr.Interface(
    fn=classify_tweet,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your tweet here..."),
        gr.Radio(choices=["Disaster", "Not Disaster"], label="Expected Classification")
    ],
    outputs=[
        gr.HTML(label=model_name) for model_name in [
           
            'Logistic Regression', 'Naive Bayes', 'XGBoost', 'SVM', 'Random Forest',
            'LLaMa2_70B', 'Falcon_7B', 'Mistral_7B'
        ]

    ]
)

# Run the Gradio app
iface.launch(inline=False)