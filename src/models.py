"""Baseline models for multi-label GO term prediction.

All models share one interface:

    model.fit(X_train, Y_train)   Y_train: sparse binary matrix (proteins x label terms)
    model.predict(X)              -> dense float32 matrix of scores in [0, 1]
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


class KmerFeaturizer:
    """Turn protein sequences into TF-IDF vectors of overlapping k-mers.

    A sequence is treated like a sentence whose "words" are all substrings of
    length k (k=3 gives up to 20^3 = 8000 features). Rows are L2-normalised,
    so the dot product of two rows is their cosine similarity.
    """

    def __init__(self, k: int = 3, min_df: int = 2):
        self.k = k
        self.vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(k, k), lowercase=False,
            sublinear_tf=True, min_df=min_df, dtype=np.float32,
        )

    def fit(self, sequences):
        self.vectorizer.fit(sequences)
        return self

    def transform(self, sequences) -> sparse.csr_matrix:
        return self.vectorizer.transform(sequences).tocsr()

    def fit_transform(self, sequences) -> sparse.csr_matrix:
        return self.vectorizer.fit_transform(sequences).tocsr()

    @property
    def n_features(self) -> int:
        return len(self.vectorizer.vocabulary_)


class NaiveFrequencyBaseline:
    """Predict the same scores for every protein: how common each term is.

    This is the classic CAFA "naive" baseline. The score of a term is the
    fraction of training proteins annotated in its aspect that carry the term.
    It ignores the sequence completely, so any useful model must beat it.
    """

    def __init__(self, label_aspects):
        self.label_aspects = np.asarray(label_aspects)

    def fit(self, X, Y):
        Y = sparse.csr_matrix(Y)
        counts = np.asarray(Y.sum(axis=0)).ravel()
        self.frequencies_ = np.zeros(Y.shape[1], dtype=np.float32)
        for aspect in np.unique(self.label_aspects):
            cols = np.where(self.label_aspects == aspect)[0]
            annotated = (np.asarray(Y[:, cols].sum(axis=1)).ravel() > 0).sum()
            self.frequencies_[cols] = counts[cols] / max(annotated, 1)
        return self

    def predict(self, X) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.tile(self.frequencies_, (n, 1))


class KNNBaseline:
    """Transfer labels from the most similar training sequences.

    Similarity = cosine between k-mer TF-IDF vectors. The score of a term is
    the similarity-weighted share of the k nearest neighbours that have it.
    This mimics the homology-transfer idea behind the BLAST baseline of CAFA
    without needing an alignment tool.
    """

    def __init__(self, k: int = 10, batch_size: int = 256):
        self.k = k
        self.batch_size = batch_size

    def fit(self, X, Y):
        self.X_train_T_ = sparse.csr_matrix(X).T.tocsr()
        self.Y_train_ = sparse.csr_matrix(Y, dtype=np.float32)
        return self

    def predict(self, X) -> np.ndarray:
        X = sparse.csr_matrix(X)
        n_train = self.Y_train_.shape[0]
        k = min(self.k, n_train)
        scores = np.zeros((X.shape[0], self.Y_train_.shape[1]), dtype=np.float32)
        for start in range(0, X.shape[0], self.batch_size):
            stop = min(start + self.batch_size, X.shape[0])
            similarity = (X[start:stop] @ self.X_train_T_).toarray()
            if k < n_train:
                neighbours = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
            else:
                neighbours = np.tile(np.arange(n_train), (stop - start, 1))
            weights = np.take_along_axis(similarity, neighbours, axis=1)
            rows = np.repeat(np.arange(stop - start), k)
            neighbour_weights = sparse.csr_matrix(
                (weights.ravel(), (rows, neighbours.ravel())), shape=(stop - start, n_train)
            )
            total = weights.sum(axis=1, keepdims=True)
            total[total == 0] = 1.0
            scores[start:stop] = (neighbour_weights @ self.Y_train_).toarray() / total
        return np.clip(scores, 0.0, 1.0)


class MLPBaseline:
    """One-hidden-layer neural network with one sigmoid output per GO term.

    Thin wrapper around scikit-learn's MLPClassifier (multi-label mode, binary
    cross-entropy loss), so it runs on CPU and needs no deep-learning framework.
    """

    def __init__(self, hidden_units: int = 512, epochs: int = 15, batch_size: int = 256,
                 learning_rate: float = 1e-3, alpha: float = 1e-5, random_state: int = 42,
                 verbose: bool = False):
        self.model = MLPClassifier(
            hidden_layer_sizes=(hidden_units,), activation="relu", solver="adam",
            alpha=alpha, batch_size=batch_size, learning_rate_init=learning_rate,
            max_iter=epochs, shuffle=True, random_state=random_state, verbose=verbose,
        )
        self.predict_batch_size = 4096

    def fit(self, X, Y):
        Y = sparse.csr_matrix(Y)
        # labels that never occur in the training split cannot be learned
        self.active_ = np.asarray(Y.sum(axis=0)).ravel() > 0
        self.n_labels_ = Y.shape[1]
        # int64 on purpose: with small integer dtypes (int8) scikit-learn's MLP
        # trained without error but produced wrong multi-label outputs in our tests
        targets = Y[:, np.where(self.active_)[0]].toarray().astype(np.int64)
        with warnings.catch_warnings():
            # stopping after a fixed number of epochs is intended here
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(X, targets)
        return self

    @property
    def loss_curve(self):
        return list(self.model.loss_curve_)

    def predict(self, X) -> np.ndarray:
        scores = np.zeros((X.shape[0], self.n_labels_), dtype=np.float32)
        active_cols = np.where(self.active_)[0]
        for start in range(0, X.shape[0], self.predict_batch_size):
            stop = min(start + self.predict_batch_size, X.shape[0])
            proba = self.model.predict_proba(X[start:stop])
            scores[start:stop, active_cols] = np.asarray(proba, dtype=np.float32).reshape(stop - start, -1)
        return scores
