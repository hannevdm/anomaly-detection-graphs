from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    average_precision_score
import numpy as np
from src.data_utils import e_load, timer
from src.embeddings import GraphSAGEModel, train_gs, get_emb_gs
from src.visualization import plot_confusion_matrix

timer(True, 'GraphSAGE and HDBSCAN')

# Load data
data = e_load()[0]

x = data.x
y = data.y

print("Generating embeddings ...")
gsmodel_if = GraphSAGEModel(in_channels= data.x.size(1), out_channels = 64)
train_gs(gsmodel_if,data)
embeddings = get_emb_gs(gsmodel_if,data,scaled=False, pca_scaled=True)
print("Embeddings generated")

# # Save embeddings
# np.save('../data/embeddings/graphsage.npy', embeddings)

# # Load embeddings
# embeddings = np.load('../data/embeddings/graphsage.npy')

# for graphsage: 100 and 1
clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=1, metric='euclidean')
y_pred = clusterer.fit_predict(embeddings)


print("dbscan")

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

print(f"GraphSAGE + HDBSCAN Accuracy: {accuracy}")
print(f"GraphSAGE + HDBSCAN Precision: {precision}")
print(f"GraphSAGE + HDBSCAN Recall: {recall}")
print(f"GraphSAGE + HDBSCAN Score: {f1}")
print(f"GraphSAGE + HDBSCAN Score: {pr_auc}")

print("Classification Report:")
print(classification_report(y_labelled, y_pred_mapped, digits=4))

timer(False, 'GraphSAGE and HDBSCAN')

# Visualization

# Plot Confusion matrix
plot_confusion_matrix(x_known.numpy(), y_labelled, y_pred_labelled, title="GraphSAGE + HDBSCAN Confusion Matrix")
