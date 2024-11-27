import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SKPCA
from PCA import PCA

# Import your PCA implementation here
# from your_pca_module import PCA

# Load the Iris dataset
data = load_iris()
X = data.data  # Feature matrix (150 samples, 4 features)


# Test your PCA implementation
def test_pca():
    # Initialize custom PCA and fit
    custom_pca = PCA()
    custom_pca.fit(X)
    X_projected_custom = custom_pca.predict(X)

    # Use sklearn's PCA for comparison
    sklearn_pca = SKPCA()
    X_projected_sklearn = sklearn_pca.fit_transform(X)

    print(X_projected_custom)
    print(X_projected_sklearn)

    # Check the shapes of the outputs
    assert X_projected_custom.shape == X_projected_sklearn.shape, \
        f"Shape mismatch: {X_projected_custom.shape} vs {X_projected_sklearn.shape}"

    # Since PCA directions may differ in sign, compare absolute values
    assert np.allclose(
        np.abs(X_projected_custom), np.abs(X_projected_sklearn), atol=1e-5
    ), "Projection mismatch between custom PCA and sklearn PCA"

    print("Custom PCA passed all tests!")


# Run the test
test_pca()
