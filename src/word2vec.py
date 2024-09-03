from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec

def get_vector(document, model):
    value = 0
    n_failer = 0
    for word in document:
        if word in model.wv:
            value += model.wv[word]
        else :
            n_failer += 1
    value /= int(len(document) - n_failer)
    return value



def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """

    if str != type(corpus):
        feature_vector = []
        for document in corpus:
            feature_vector.append(get_vector(document, model))
        return np.array(feature_vector)
    else:
        return get_vector(corpus, model)
