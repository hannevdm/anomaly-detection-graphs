from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    average_precision_score
import numpy as np
from src.data_utils import e_load, timer
from src.embeddings import Node2VecModel, train_n2v, get_emb_n2v

# Load dataset
from src.visualization import plot_scores, plot_confusion_matrix
timer(True, 'Node2Vec and HDBSCAN')
data = e_load()[0]

x = data.x
y = data.y

# Embeddings
print("Generating embeddings ...")
n2vmodel_if = Node2VecModel(data.edge_index, embedding_dim=128, walk_length=120, context_size=10, walks_per_node=100,num_negative_samples=1,
                            p=2, q=0.5)
train_n2v(n2vmodel_if, 5)
embeddings = get_emb_n2v(n2vmodel_if, scaled=False, pca_scaled=True)
print("Embeddings generated")

# Save embeddings
#np.save('../data/embeddings/node2vec.npy', embeddings)

# Load embeddings
#embeddings = np.load('../data/embeddings/node2vec.npy')


clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1, metric='euclidean')
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

print(f"Node2Vec + HDBSCAN Accuracy: {accuracy}")
print(f"Node2Vec + HDBSCAN Precision: {precision}")
print(f"Node2Vec + HDBSCAN Recall: {recall}")
print(f"Node2Vec + HDBSCAN Score: {f1}")
print(f"Node2Vec + HDBSCAN Score: {pr_auc}")


print("Classification Report:")
print(classification_report(y_labelled, y_pred_mapped, digits=4))
timer(False, 'Node2Vec and HDBSCAN')

# Visualization

# Plot Confusion matrix
plot_confusion_matrix(x_known.numpy(), y_labelled, y_pred_labelled, title="Node2Vec + HDBSCAN Confusion Matrix")
