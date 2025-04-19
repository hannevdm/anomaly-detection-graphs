import hdbscan
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, \
    classification_report

from src.data_utils import timer, e_load

timer(True, 'HDBSCAN raw')


dataset = e_load()
data = dataset[0]

x = data.x.cpu().numpy()
y = data.y.cpu().numpy()

pca = PCA(n_components=2)
x_scaled = pca.fit_transform(x)

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric='euclidean')
y_pred = clusterer.fit_predict(x_scaled)


# Keep only labelled nodes
mask_labelled = y != 2
y_labelled = y[mask_labelled]
y_pred_labelled = y_pred[mask_labelled]
x_known = x[mask_labelled]

y_pred_mapped = np.where(y_pred_labelled == -1, 1, 0)

# Evaluation
accuracy = accuracy_score(y_labelled, y_pred_mapped)
precision = precision_score(y_labelled, y_pred_mapped)
recall = recall_score(y_labelled, y_pred_mapped)
f1 = f1_score(y_labelled, y_pred_mapped)
pr_auc = average_precision_score(y_labelled, y_pred_mapped)

print(f"raw HDBSCAN Accuracy: {accuracy}")
print(f"raw HDBSCAN Precision: {precision}")
print(f"raw HDBSCAN Recall: {recall}")
print(f"raw HDBSCAN Score: {f1}")
print(f"raw HDBSCAN Score: {pr_auc}")

print("Classification Report:")
print(classification_report(y_labelled, y_pred_mapped, digits=4))

timer(False, 'HDBSCAN raw')