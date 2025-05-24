import csv
from matplotlib import pyplot as plt
import hdbscan
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    average_precision_score
import numpy as np
from sklearn.model_selection import ParameterSampler
from src.data_utils import e_load, timer, save_anomaly_scores
from src.embeddings import GraphSAGEModel, train_gs, get_emb_gs, embeddings_grid, models_grid
from src.visualization import plot_confusion_matrix

embedding_method = 'GraphSAGE'
model = 'HDBSCAN'

timer(True, f'{embedding_type} and {model}')

# Load data
data = e_load()[0]

x = data.x
y = data.y

# Grid search

# best_f1 = float('-inf')
# csv_file = '../data/configs/results_{embedding_type}_{model}.csv'
# headers = ['Embedding Params', 'Model Params', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'PR-AUC']
# embeddings_cache = {}

# # parameter search
# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(headers)
#
#     for emb_params in ParameterSampler(embeddings_grid[embedding_method], n_iter=10, random_state=42):
#         emb_key = str(sorted(emb_params.items()))
#         if emb_key in embeddings_cache:
#             embeddings = embeddings_cache[emb_key]
#             print("reusing embeddings")
#         else:
#             # Build and train embedding model
#             gsmodel = GraphSAGEModel(
#                 in_channels=data.x.size(1),
#                 out_channels=emb_params['out_channels'],
#                 num_layers=emb_params['layers'],
#                 hidden_channels=emb_params['hidden_channels'],
#                 learning_rate=emb_params['learning_rate']
#             )
#             train_gs(gsmodel, data, epochs=emb_params['epochs'])
#             embeddings = get_emb_gs(gsmodel, data, scaled=False, pca_scaled=True)
#             embeddings_cache[emb_key] = embeddings
#
#         for model_params in ParameterSampler(models_grid[model_type], n_iter=2, random_state=42):
#             print(f"\nRunning {embedding_method} + {model_type} with:")
#             print(f"Embedding params: {emb_params}")
#             print(f"Model params: {model_params}")
#
#             # Apply HDBSCAN
#             clusterer = hdbscan.HDBSCAN(
#                 min_cluster_size=model_params['min_cluster'],
#                 min_samples=model_params['min_samples'],
#                 metric='euclidean'
#             )
#             y_pred = clusterer.fit_predict(embeddings)
#
#             # Filter out label 2 (unknown)
#             mask_labelled = y != 2
#             y_labelled = y[mask_labelled]
#             y_pred_labelled = y_pred[mask_labelled]
#
#             # Map cluster output: noise (-1) = fraud (1), else normal (0)
#             y_pred_mapped = np.where(y_pred_labelled == -1, 1, 0)
#
#             # Compute metrics
#             acc = accuracy_score(y_labelled, y_pred_mapped)
#             prec = precision_score(y_labelled, y_pred_mapped)
#             rec = recall_score(y_labelled, y_pred_mapped)
#             f1 = f1_score(y_labelled, y_pred_mapped)
#             pr = average_precision_score(y_labelled, y_pred_mapped)
-#
#             # Write the results to the CSV file
#             result = [
#                 str(emb_params), str(model_params), acc, prec, rec, f1, pr
#             ]
#             with open(csv_file, mode='a', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerow(result)
#
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_params = {
#                     'embedding': emb_params,
#                     'model': model_params
#                 }
#                 print(best_params)


print("Generating embeddings ...")
gsmodel_if = GraphSAGEModel(in_channels= data.x.size(1), out_channels = 64)
train_gs(gsmodel_if,data)
embeddings = get_emb_gs(gsmodel_if,data,scaled=False, pca_scaled=True)
print("Embeddings generated")

# # Save embeddings
# np.save('../data/embeddings/{model}_{embedding_type}.npy', embeddings)

# # Load embeddings
# embeddings = np.load('../data/embeddings/{model}_{embedding_type}.npy')

# for graphsage: 100 and 1
clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=1, metric='euclidean')
y_pred = clusterer.fit_predict(embeddings)

# Keep only labelled nodes
mask_labelled = y != 2
y_labelled = y[mask_labelled]
y_pred_labelled = y_pred[mask_labelled]
x_known = x[mask_labelled]
labelled_indices = mask_labelled.nonzero(as_tuple=True)[0]
embeddings_labelled = embeddings[labelled_indices]
scores = clusterer.outlier_scores_
scores_known = scores[labelled_indices]

# Convert DBSCAN output to fraud prediction:
# We'll treat noise points (-1) as fraud (1), and others as normal (0)
y_pred_mapped = np.where(y_pred_labelled == -1, 1, 0)

# Evaluation
accuracy = accuracy_score(y_labelled, y_pred_mapped)
precision = precision_score(y_labelled, y_pred_mapped)
recall = recall_score(y_labelled, y_pred_mapped)
f1 = f1_score(y_labelled, y_pred_mapped)
pr_auc = average_precision_score(y_labelled, y_pred_mapped)

print(f"{embedding_type} + {model} Accuracy: {accuracy}")
print(f"{embedding_type} + {model} Precision: {precision}")
print(f"{embedding_type} + {model} Recall: {recall}")
print(f"{embedding_type} + {model} Score: {f1}")
print(f"{embedding_type} + {model} Score: {pr_auc}")

print("Classification Report:")
print(classification_report(y_labelled, y_pred_mapped, digits=4))

timer(False, 'GraphSAGE and HDBSCAN')

# Visualization

# Plot Confusion matrix
plot_confusion_matrix(x_known.numpy(), y_labelled, y_pred_labelled, title=f"{embedding_type} + {model} Confusion Matrix")

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
save_anomaly_scores(scores_known, y_labelled, f"{model}", f"{embedding_type}")
