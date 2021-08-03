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
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import tensorflow_hub as hub


def main():
    with open(os.path.join(os.path.dirname(__file__), 'titles.txt')) as fin:
        titles = [x.strip() for x in fin.readlines() if x.strip()]
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/4")
    batch_size = 100
    title_embeddings = []
    for i in range(0, len(titles), batch_size):
        tensor_result = embed(titles[i:i + batch_size])['outputs']
        title_embeddings.append(tensor_result.numpy())
    title_embeddings = np.concatenate(title_embeddings)
    nearest_neighbour = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1,
        )
    nearest_neighbour.fit(title_embeddings, range(1, 10_001))
    joblib.dump(nearest_neighbour, os.path.join(os.path.dirname(__file__), 'nearest_neighbour.jld'))


if __name__ == '__main__':
    main()
