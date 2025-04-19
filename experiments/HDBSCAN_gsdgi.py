from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    average_precision_score
import numpy as np
from src.data_utils import e_load, timer
from src.embeddings import get_dgi_embeddings, train_dgi, create_dgi_model

# Load dataset
from src.visualization import plot_confusion_matrix

timer(True, 'GraphSAGE DGI and HDBSCAN')

data = e_load()[0]

x = data.x
y = data.y

print("Generating embeddings ...")
dgi_model = create_dgi_model(in_channels=data.x.size(1), hidden_channels=32, out_channels=32)
dgi_model = train_dgi(dgi_model, data, epochs=150)
embeddings = get_dgi_embeddings(dgi_model, data, scaled=False, pca_scaled=True)
print("Embeddings generated")

# # Save embeddings
# np.save('../data/embeddings/graphsageDGI.npy', embeddings)

# # Load embeddings
# embeddings = np.load('../data/embeddings/graphsageDGI.npy')

clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=1, metric='euclidean')
y_pred = clusterer.fit_predict(embeddings)


# Keep only labelled nodes
mask_labelled = y != 2
y_labelled = y[mask_labelled]
y_pred_labelled = y_pred[mask_labelled]
x_known = x[mask_labelled]

# Convert DBSCAN output to fraud prediction:
# We'll treat noise points (-1) as fraud (1), and others as normal (0)
y_pred_mapped = np.where(y_pred_labelled == -1, 1, 0)

# Evaluation
accuracy = accuracy_score(y_labelled, y_pred_mapped)
precision = precision_score(y_labelled, y_pred_mapped)
recall = recall_score(y_labelled, y_pred_mapped)
f1 = f1_score(y_labelled, y_pred_mapped)
pr_auc = average_precision_score(y_labelled, y_pred_mapped)

print(f"GraphSAGE DGI + HDBSCAN Accuracy: {accuracy}")
print(f"GraphSAGE DGI + HDBSCAN Precision: {precision}")
print(f"GraphSAGE DGI + HDBSCAN Recall: {recall}")
print(f"GraphSAGE DGI + HDBSCAN Score: {f1}")
print(f"GraphSAGE DGI + HDBSCAN Score: {pr_auc}")

print("Classification Report:")
print(classification_report(y_labelled, y_pred_mapped, digits=4))

timer(False, 'GraphSAGE DGI and HDBSCAN')

# Visualization

# Plot Confusion matrix
plot_confusion_matrix(x_known.numpy(), y_labelled, y_pred_labelled, title="GraphSAGE DGI + HDBSCAN Confusion Matrix")
