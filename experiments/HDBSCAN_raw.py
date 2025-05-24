import csv
import hdbscan
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, \
    classification_report
from sklearn.model_selection import ParameterGrid
from src.embeddings import models_grid
from src.data_utils import timer, e_load, save_anomaly_scores

embedding_method = 'Raw features'
model = 'HDBSCAN'

timer(True, f'{embedding_method} and {model}')

data = e_load()[0]

x = data.x.cpu().numpy()
y = data.y.cpu().numpy()

pca = PCA(n_components=2)
x_scaled = pca.fit_transform(x)

# # Grid search
# csv_file = f'../data/configs/results_{embedding_method}_{model}.csv'
# headers = ['Model Params', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'PR-AUC']
# best_f1 = float('-inf')

# # parameter search
# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(headers)

#     for model_params in ParameterGrid(models_grid[model_type]):
#         print(f"Model params: {model_params}")
#         clusterer = hdbscan.HDBSCAN(min_cluster_size=model_params['min_cluster'], min_samples=model_params['min_samples'], metric='euclidean')
#         y_pred = clusterer.fit_predict(x_scaled)

#         # Keep only labelled nodes
#         mask_labelled = y != 2
#         y_labelled = y[mask_labelled]
#         y_pred_labelled = y_pred[mask_labelled]
#         x_known = x[mask_labelled]

#         y_pred_mapped = np.where(y_pred_labelled == -1, 1, 0)

#         # Evaluation
#         accuracy = accuracy_score(y_labelled, y_pred_mapped)
#         precision = precision_score(y_labelled, y_pred_mapped)
#         recall = recall_score(y_labelled, y_pred_mapped)
#         f1 = f1_score(y_labelled, y_pred_mapped)
#         pr_auc = average_precision_score(y_labelled, y_pred_mapped)

#         # Write the results to the CSV file
#         result = [
#             str(model_params), accuracy, precision, recall, f1, pr_auc
#         ]

#         with open(csv_file, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(result)

#         if f1 > best_f1:
#             best_f1 = f1
#             best_params = model_params

#             print(best_params)


clusterer = hdbscan.HDBSCAN(min_cluster_size=60, min_samples=2, metric='euclidean')
y_pred = clusterer.fit_predict(x_scaled)

# Keep only labelled nodes
mask_labelled = y != 2
y_labelled = y[mask_labelled]
y_pred_labelled = y_pred[mask_labelled]
x_known = x[mask_labelled]
y_pred_mapped = np.where(y_pred_labelled == -1, 1, 0)
labelled_indices = mask_labelled.nonzero(as_tuple=True)[0]
scores = clusterer.outlier_scores_
scores_known = scores[labelled_indices]

# Evaluation
accuracy = accuracy_score(y_labelled, y_pred_mapped)
precision = precision_score(y_labelled, y_pred_mapped)
recall = recall_score(y_labelled, y_pred_mapped)
f1 = f1_score(y_labelled, y_pred_mapped)
pr_auc = average_precision_score(y_labelled, y_pred_mapped)

print(f"{embedding_method} and {model} Accuracy: {accuracy}")
print(f"{embedding_method} and {model} Precision: {precision}")
print(f"{embedding_method} and {model} Recall: {recall}")
print(f"{embedding_method} and {model} Score: {f1}")
print(f"{embedding_method} and {model} Score: {pr_auc}")

print("Classification Report:")
print(classification_report(y_labelled, y_pred_mapped, digits=4))

timer(False, f'{embedding_method} and {model}')

# Anomaly Score Distribution
plt.hist(scores_known[y_labelled == 0], bins=50, alpha=0.6, label="Licit")
plt.hist(scores_known[y_labelled == 1], bins=50, alpha=0.6, label="Fraud")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.title("Anomaly Score Distribution")
plt.legend()
plt.tight_layout()
plt.show()

# Export scores
save_anomaly_scores(scores_known, y_labelled, f"{model}", f"{embedding_method}")
