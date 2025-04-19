import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from umap import UMAP

def plot_loss(losses, name):
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{str(name)} Training Loss")
    plt.show()

def plot_pca_embeddings(pca_embeddings, labels):
    plt.scatter(pca_embeddings[:len(labels), 0], pca_embeddings[:len(labels), 1], c=labels,
                cmap='coolwarm', alpha=0.5)
    plt.colorbar(label="Class (0 = Normal, 1 = Fraud)")
    plt.title("PCA Visualization of Labelled Node Embeddings")
    plt.show()

def plot_anomaly_scores(scores, title="Distribution of Scores"):
    plt.hist(scores, bins=50)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()

def plot_anomaly_scores_vs_predicted_fraud(scores, frauds, title="Distribution of Anomaly Scores and Predicted Fraud"):
    plt.hist(frauds, bins=30, alpha=0.7, label="Fraud", color='red')
    plt.hist(scores, bins=30, alpha=0.7, label="Normal", color='blue')
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_scores(labels, scores, title =""): # only plots labelled data
    plt.hist(scores[labels == 0], bins=50, alpha=0.6, label="Licit")
    plt.hist(scores[labels == 1], bins=50, alpha=0.6, label="Fraud")
    plt.axvline(np.percentile(scores, 80), color='r', linestyle='--', label='80th percentile')
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title(f"{title} Anomaly Score Distribution")
    plt.legend()
    plt.show()

def plot_confusion_matrix(embeddings, true_labels, y_pred, title="UMAP Embedding + Predictions", map= True):
    """
    Visualize embeddings using UMAP with color-coded IF confusion matrix labels:
    TP = green, FN = yellow, FP = red, TN = blue
    """
    # Convert IF predictions: -1 → anomaly (illicit) → label 1; 1 → inlier (licit) → label 0
    if map:
        y_pred_mapped = np.where(y_pred == -1, 1, 0)
    else: y_pred_mapped = y_pred

    # Reduce dimensions
    reducer = UMAP(n_components=2, random_state=42)
    X_reduced = reducer.fit_transform(embeddings)

    # Assign confusion matrix colors
    colors = []
    for true, pred in zip(true_labels, y_pred_mapped):
        if true == 1 and pred == 1:
            colors.append("green")   # TP: Illicit correctly identified
        elif true == 1 and pred == 0:
            colors.append("yellow")  # FN: Illicit missed
        elif true == 0 and pred == 1:
            colors.append("red")     # FP: Licit wrongly flagged
        else:
            colors.append("blue")    # TN: Licit correctly identified

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=colors, s=10, alpha=0.8)
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True)
    plt.show()