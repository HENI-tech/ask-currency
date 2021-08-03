# Copyright 2021 HENI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

import joblib
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import tensorflow_hub as hub


app = FastAPI()


class AnswerResponse(BaseModel):
    answer: str
    number: int


def load_model():
    path = os.path.join(os.path.dirname(__file__), 'nearest_neighbour.jld')
    nn = joblib.load(path)
    return nn


with open(os.path.join(os.path.dirname(__file__), 'titles.txt')) as fin:
    TITLES = [x.strip() for x in fin.readlines() if x.strip()]


NN = load_model()
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/4")


@app.get('/', response_model=AnswerResponse)
def ask_currency(question: str):
    question = question.strip()
    if len(question) > 1000:
        question = question[:1000]
    question_embedding = embed([question])['outputs']
    question_embedding = question_embedding.numpy()
    probabilities = NN.predict_proba(question_embedding)

    idx = np.random.choice(10_000, p=probabilities[0])
    title = TITLES[idx]
    title_number = idx + 1
    return {'answer': title, 'number': title_number}
