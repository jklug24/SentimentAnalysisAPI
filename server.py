from BernoulliTextModel import model
import pandas as pd
from typing import Union
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

description = """
The Airline Sentiment Analysis API uses a Naive Bayes model trained on airline comments to predict the sentiment of text.

## Endpoints

You can:
* Predict sentiment of text
"""

df = pd.read_csv('airline_sentiment_analysis.csv')
df = df.drop('Unnamed: 0', axis=1)
X = df.text.to_numpy()
y = df.airline_sentiment.to_numpy()
m = model()
m.train(X, y)

tagMetadata = [
    {
        'name': 'Use',
        'description': 'endpoints that are exposed for general use of the model'
    }
]

app = FastAPI(    
    title="Airline Sentiment Analysis API",
    description=description,
    version="0.0.1",
    contact={
        "name": "Joseph Klug",
        "email": "josephmklug@gmail.com",
    }
)


@app.get("/predict", tags=['Use'])
def read_item(text: Union[str, None] = None):
    if (str != None):
        return {"prediction": m.predict(text)}
    else: return {}