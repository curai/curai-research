import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

# TODO combine into a single pipeline that can be imported

INPUT_FEATURES = ["prev_dr_turn", "prev_pt_resp", "curr_dr_turn"]

class EmbedTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        embedding_model = 'paraphrase-mpnet-base-v2'
        model = SentenceTransformer(embedding_model)
        model.max_seq_length = 512
        self.embedding_len = len(model.encode("test"))
        print("Embedding length:", self.embedding_len)
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create the embeddings
        X_copy = X.copy()
        for name in INPUT_FEATURES:
            X_copy[f"{name}_emb"] = X[name].progress_map(self.model.encode)
        
        # Expand out and combine the embeddings into a single dataframe
        split_cols = [
            pd.DataFrame(X_copy[f"{name}_emb"].to_list(), columns=[f"{name}_feat{i + 1}" for i in range(self.embedding_len)]) 
                for name in INPUT_FEATURES]

        # Concatenate them all into one giant 768 * 3 column wide DF
        concat = pd.concat(split_cols, axis=1)

        return concat

class MessagePCA(BaseEstimator, TransformerMixin):
    """
    We apply PCA to each message separately, rather than the entire input at once,
    hence the custom wrapper class
    """

    def __init__(self, n_components = 20):
        self.n_components = n_components
        self.pca_fit = None
    
    def fit(self, X, y=None):
        self.pca_fit = PCA(n_components = self.n_components, random_state=42).fit(X)
        return self

    def transform(self, X):
        return self.pca_fit.transform(X)
