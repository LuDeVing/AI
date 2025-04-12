# 🤖 AI & Machine Learning Algorithms 💡

Welcome to the **AI & Machine Learning Algorithms** repository!  
This project showcases 🛠️ implementations of various fundamental ML algorithms in Python — from simple 🔁 linear regression to powerful ensemble methods like AdaBoost and Neural networks. Each algorithm is self-contained in its own file, with shared helper functions where needed. Each algorithm is implemented from scratch with minimal helper libraries, Although some of the tests are written by AI.

📦 Includes example scripts to help you **train**, **test**, and **evaluate** some of the models.

---

## 📚 Table of Contents

1. 🧭 [Overview](#overview)  
2. 🧠 [Features and Implemented Algorithms](#features-and-implemented-algorithms)  
3. 🗂️ [Project Structure](#project-structure)  
4. 📦 [Dependencies and Installation](#dependencies-and-installation)  
5. 🧪 [Examples](#examples)  

---

## Overview

The goal of this project is to provide 👀 **clear, educational** implementations of popular machine learning algorithms. Use these as a:

- 📘 **Learning resource**
- 🧪 **Playground for experimentation**
- 🧱 **Starter kit for your own AI projects**

Features include:

✅ Step-by-step logic  
✅ Minimal use of third-party black-box tools  
✅ Tweak-friendly parameters  

---

## Features and Implemented Algorithms

| 🧠 Algorithm                | 📄 File Name                    | 🔍 Description |
|----------------------------|----------------------------------|----------------|
| 📉 Linear Regression       | `Linear_Regression.py`                      | Basic Linear regression |
| 📈 Logistic Regression     | `Logistic_Regression.py`                    | Binary Logistic. |
| 🧮 Naive Bayes             | `Naive_Bayes.py` / `Multinomial_Naive_Bayes.py` | Gaussian & multinomial versions. Great for text! |
| 🌳 Decision Tree           | `Decision_Tree.py`                          | Splits based on Gini entropy. Easy to visualize! |
| 🌲 Random Forest           | `RandomForest.py`                           | Forest of Decision trees, great for accuracy. |
| 💥 AdaBoost                | `AdaBoost.py`                               | Focuses on tricky cases with adaptive weighting. |
| 🧬 Neural Network          | `Neural_Network.py` + `Neural_Functions.py` | Simple Neural network implementation using numpy with helper loss functions|
| ⚖️ Support Vector Machine  | `SVM.py` / `SVM_Kernel.py`                  | Linear & kernelized (RBF). |
| 🎯 K-Means Clustering      | `K_Means_Clustering.py`                     | Classic unsupervised grouping of similar points. |
| 📉 PCA                     | `PCA.py`                                    | Reduces data dimensionality while preserving variance. |
| 🧾 MultiHead Attention     | `MultiHeadAttention.py`                     | MultiHead Attention implementation for generative transformers |

---


## Project Structure
```

├── .idea/                     # IDE configuration files (optional)
├── AdaBoost.py                # AdaBoost implementation
├── Decision_Tree.py           # Decision tree classifier
├── K_Means_Clustering.py      # K-means clustering
├── Linear_Regression.py       # Linear regression
├── Logistic_Regression.py     # Logistic regression
├── MultiHeadAttention.py      # Example of multihead atention
├── Multinomial_Naive_Bayes.py # Multinomial Naive Bayes classifier
├── Naive_Bayes.py             # General Naive Bayes classifier
├── Neural_Functions.py        # Loss functions for neural network
├── Neural_Network.py          # Simple neural network
├── PCA.py                     # Principal Component Analysis
├── RandomForest.py            # Random Forest classifier
├── README.md                  # This README
├── SVM.py                     # Support Vector Machine
├── SVM_Kernel.py              # Kernelized SVM
├── main.py                    # Sample main entry point (optional)
└── test.py                    # Scripts for testing algorithms

```

## Dependencies and Installation

1. **Clone the repository**:
```
git clone https://github.com/YOUR_USERNAME/AI-ML-Algorithms.git
```

2. **Create a virtual environment (recommended)**:
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install required libraries**:
```
# tensoflow is only used in multihead attention 
pip install numpy tensorflow scikit-learn matplotlib
```

## Examples

- Bellow is a code snippet of **NeuralNetwork**-s utalization:

```
# Importing
from Neural_Network import NeuralLayer, ActivationLayer, NeuralNetwork

# Instanciating neural network with final entropy function with it's differential
nn = NeuralNetwork(loss=cross_entropy_loss, loss_diff=cross_entropy_loss_diff, learning_rate=0.7, iters=2000, display_loss=True)

# Adding layers to neural network
nn.add_layer(NeuralLayer(784, 16))  # Input layer to hidden layer
nn.add_layer(ActivationLayer(relu, relu_diff))
nn.add_layer(NeuralLayer(16, 16))   # Hidden layer to hidden layer
nn.add_layer(ActivationLayer(relu, relu_diff))
nn.add_layer(NeuralLayer(16, 10))    # Hidden layer to output layer
nn.add_layer(ActivationLayer(softmax, lambda x: 1))  # Output layer with softmax activation

# Fitting the network
nn.fit(X_train, y_train)

# Predicting the output
y_pred = nn.predict(X_test)

```
- Below is a snippet demonstrating how to use the **Logistic Regression**:

```
from Logistic_Regression import LogisticRegression
import numpy as np

# Example dataset
X = np.array([[0.5, 1.2], [1.0, 2.3], [1.5, 3.0], [2.0, 4.1]])
y = np.array([0, 0, 1, 1])

# Initialize and train the logistic regression model
model = LogisticRegression(lr=0.1, iters=1000)
model.fit(X, y)

# Predict on the training data
predictions = model.predict(X)
print("Predictions:", predictions)

# Evaluate accuracy
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)
```

Replicate a similar approach for other algorithms in the repository for use.

---

Made with ❤️ for learning and sharing.
