import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SKPCA
from PCA import PCA

data = load_iris()
X = data.data


def test_pca():
    custom_pca = PCA()
    custom_pca.fit(X)
    X_projected_custom = custom_pca.predict(X)

    sklearn_pca = SKPCA()
    X_projected_sklearn = sklearn_pca.fit_transform(X)

    print(X_projected_custom)
    print(X_projected_sklearn)

    assert X_projected_custom.shape == X_projected_sklearn.shape, \
        f"Shape mismatch: {X_projected_custom.shape} vs {X_projected_sklearn.shape}"

    assert np.allclose(
        np.abs(X_projected_custom), np.abs(X_projected_sklearn), atol=1e-5
    ), "Projection mismatch between custom PCA and sklearn PCA"

    print("Custom PCA passed all tests!")


test_pca()
