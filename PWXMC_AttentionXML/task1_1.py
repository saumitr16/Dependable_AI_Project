import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from lime.lime_text import LimeTextExplainer
import os
import torch
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Paths
data_dir = Path("data/EUR-Lex")
results_dir = Path("explainations")
results_dir.mkdir(exist_ok=True)

# Load data
train_texts = np.load(data_dir / "train_texts.npy", allow_pickle=True)
test_texts = np.load(data_dir / "test_texts.npy", allow_pickle=True)
train_labels = np.load(data_dir / "train_labels.npy", allow_pickle=True)
test_labels = np.load(data_dir / "test_labels.npy", allow_pickle=True)

# Convert texts to list of strings if necessary
train_texts = [str(text) for text in train_texts]
test_texts = [str(text) for text in test_texts]

# Convert labels to binary array format
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_labels)
test_labels = mlb.transform(test_labels)

# Save the MultiLabelBinarizer
with open(results_dir / "mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)

# Vectorize texts
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Train one-vs-rest logistic regression
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000), n_jobs=-1)
clf.fit(X_train, train_labels)

# Save the model and vectorizer
with open(results_dir / "ovr_logreg_model.pkl", "wb") as f:
    pickle.dump(clf, f)
with open(results_dir / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# LIME Explainer
explainer = LimeTextExplainer(class_names=[str(i) for i in range(train_labels.shape[1])])

# Function to predict probabilities for LIME
def predict_proba(texts):
    X = vectorizer.transform(texts)
    return clf.predict_proba(X)

# Integrated Gradients approximation (simplified for logistic regression)
def integrated_gradients(text, label_idx, baseline=None, steps=50):
    if baseline is None:
        baseline = vectorizer.transform([""]).toarray()[0]
    X = vectorizer.transform([text]).toarray()[0]
    interpolated = np.array([baseline + (float(i) / steps) * (X - baseline) for i in range(steps)])
    estimator = clf.estimators_[label_idx]
    grads = []
    for x in interpolated:
        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        coef = torch.tensor(estimator.coef_, dtype=torch.float32)
        bias = torch.tensor(estimator.intercept_, dtype=torch.float32)
        logit = torch.sum(coef * x_torch) + bias
        logit.backward()
        grads.append(x_torch.grad.numpy())
    avg_grads = np.mean(grads, axis=0)
    contrib = (X - baseline) * avg_grads
    return contrib

# Generate explanations for a subset of test instances
num_samples = 10
lime_explanations = []
ig_explanations = []
feature_names = vectorizer.get_feature_names_out()

for idx in range(min(num_samples, len(test_texts))):
    text = test_texts[idx]
    true_labels = np.where(test_labels[idx] == 1)[0]
    if len(true_labels) == 0:
        continue
    label_idx = true_labels[0]  # Explain the first positive label

    # LIME explanation
    lime_exp = explainer.explain_instance(
        text, predict_proba, num_features=10, labels=[label_idx]
    )
    lime_explanations.append({
        "idx": idx,
        "text": text,
        "label": label_idx,
        "explanation": lime_exp.as_list(label=label_idx)
    })

    # Integrated Gradients
    ig_contrib = integrated_gradients(text, label_idx)
    ig_top_features = [(feature_names[i], ig_contrib[i]) for i in np.argsort(-np.abs(ig_contrib))[:10]]
    ig_explanations.append({
        "idx": idx,
        "text": text,
        "label": label_idx,
        "explanation": ig_top_features
    })

# Save explanations
with open(results_dir / "lime_explanations.pkl", "wb") as f:
    pickle.dump(lime_explanations, f)
with open(results_dir / "ig_explanations.pkl", "wb") as f:
    pickle.dump(ig_explanations, f)

print(f"Explanations saved to {results_dir}")