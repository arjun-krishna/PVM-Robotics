from sklearn.decomposition import PCA as sk_PCA # noqa

class Identity:
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "identity"
    
    def fit(self, X):
        return self

    def fit_transform(self, X):
        return X
    
    def transform(self, X):
        return X

    def reduced_dim(self, input_dim):
        return input_dim


class PCA(sk_PCA):
    def reduced_dim(self, _):
        return self.n_components
    