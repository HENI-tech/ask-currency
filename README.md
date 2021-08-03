# Ask Currency API

This is a simple implementation of an API that powers the Ask Currency bot in HENI's discord server.
Join the HENI Discord Server to see it in action! https://discord.com/invite/98SrzyMmbZ

![One of the 10,000 unique Tenders by Damien Hirst, collectively called The Currency](https://dv7mrxymjpv22.cloudfront.net/1024px/1234%20Front.png)

All of the titles are embedded into a vector space using the Universal Sentence Encoder from Tensorflow hub (https://tfhub.dev/google/universal-sentence-encoder-large/4).
A KNeighborsClassifier is then built using scikit-learn (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
The only change to the defaults for this model is `weights='distance'`.

When a question comes in it is embedded into the same vector space using the Universal Sentence Encoder.
The nearest neighbors in the vector space are then found using the KNeighborsClassifer.
At this point there are 5 possibilities from which to sample from.
The chance of one of the 5 titles being picked is increased if the title embedding is closer to the question embedding.

## Setup

This API requires python3 to be installed. To keep things on your machine tidy you may want to use a virtualenv.
Due to the large memory footprint of the embedding model, the API will require a little more than 2GB RAM.
To install the required python modules and calculate the title embeddings, change directory to the root of this repo and run the following command:

```
./setup.sh
```

This step will take a while but is only needed once. It installs all python dependencies and calculates the embeddings of the 10,000 titles.

## Running the API

To then start the server:

```
./init.sh
```

Navigate to http://127.0.0.1:8000/docs in your browser to try it out.
The first request will be quite slow as the tensorflow embedding model takes a while to initialise.
Subsequent requests should be handled quite quickly.

## Usage from python

To use the deployed API from python in another terminal on your local machine:

```
>>> import requests
>>> resp = requests.get('http://127.0.0.1:8000', params={'question': 'how are you?'})
>>> resp.json()
{'answer': "How well you're doing", 'number': 7504}
```

To load and use the model from python directly:

```
>>> from ask_currency.api import ask_currency
>>> result = ask_currency('how are you?')
>>> result
{'answer': "How well you're doing", 'number': 7504}
```
