"""
Custom sklearn transformers shared by training and inference.
Must be importable before loading aqi_pipeline.pkl with joblib.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to [lower, upper] percentile computed on training data."""

    def __init__(self, upper: float = 0.999, lower: float = 0.0):
        self.upper = upper
        self.lower = lower

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.high_ = np.nanpercentile(X, self.upper * 100, axis=0)
        self.low_ = np.nanpercentile(X, self.lower * 100, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.low_, self.high_)
