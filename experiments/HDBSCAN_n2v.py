import csv
import hdbscan
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    average_precision_score
import numpy as np
from src.data_utils import e_load, timer, save_anomaly_scores
from src.embeddings import Node2VecModel, train_n2v, get_emb_n2v, embeddings_grid, models_grid
from sklearn.model_selection import ParameterGrid
from src.visualization import plot_scores, plot_confusion_matrix

embedding_method = 'Node2Vec'
model = 'HDBSCAN'

timer(True, f'{embedding_method} and {model}')

# Load dataset
data = e_load()[0]

x = data.x
y = data.y

# # Grid Search
# best_f1 = float('-inf')
# csv_file = f'../data/configs/results_{embedding_method}_{model}.csv'
# headers = ['Embedding Params', 'Model Params', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'PR-AUC']
# embeddings_cache = {}

# # Open the file in write mode to write the headers
# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(headers)

#     for emb_params in ParameterGrid(embeddings_grid[embedding_method]):
#         emb_key = str(sorted(emb_params.items()))
#         if emb_key in embeddings_cache:
#             embeddings = embeddings_cache[emb_key]
#             print("reusing embeddings")
#         else:
#             # Build and train embedding model
#             n2vmodel_if = Node2VecModel(
#                 data.edge_index,
#                 embedding_dim=emb_params['emb_dim'],
#                 walk_length=emb_params['walk_length'],
#                 walks_per_node=emb_params['walks'],
#                 context_size=emb_params['context_size'],
#                 num_negative_samples=emb_params['samples'],
#                 p=emb_params['p'],
#                 q=emb_params['q']
#             )

#             train_n2v(n2vmodel_if, epochs=emb_params['epochs'])
#             embeddings = get_emb_n2v(n2vmodel_if, scaled=False, pca_scaled=False)
#             embeddings_cache[emb_key] = embeddings

#         for model_params in ParameterGrid(models_grid[model_type]):
#             print(f"\nRunning {embedding_method} + {model_type} with:")
#             print(f"Embedding params: {emb_params}")
#             print(f"Model params: {model_params}")

#             clusterer = hdbscan.HDBSCAN(
#                 min_cluster_size=model_params['min_cluster'],
#                 min_samples=model_params['min_samples'],
#                 metric='euclidean'
#             )
#             y_pred = clusterer.fit_predict(embeddings)

#             # Filter out label 2 (unknown)
#             mask_labelled = y != 2
#             y_labelled = y[mask_labelled]
#             y_pred_labelled = y_pred[mask_labelled]

#             # Map cluster output: noise (-1) = fraud (1), else normal (0)
#             y_pred_mapped = np.where(y_pred_labelled == -1, 1, 0)

#             # Compute metrics
#             acc = accuracy_score(y_labelled, y_pred_mapped)
#             prec = precision_score(y_labelled, y_pred_mapped)
#             rec = recall_score(y_labelled, y_pred_mapped)
#             f1 = f1_score(y_labelled, y_pred_mapped)
#             pr = average_precision_score(y_labelled, y_pred_mapped)

#             # Write the results to the CSV file
#             result = [
#                 str(emb_params), str(model_params), acc, prec, rec, f1, pr
#             ]
#             with open(csv_file, mode='a', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerow(result)

#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_params = {
#                     'embedding': emb_params,
#                     'model': model_params
#                 }
#                 print(best_params)


# Embeddings
print("Generating embeddings ...")
n2vmodel_hdbscan = Node2VecModel(data.edge_index, embedding_dim=512, walk_length=120, context_size=15, walks_per_node=100,num_negative_samples=2,
                            p=0.5, q=2)
train_n2v(n2vmodel_hdbscan, 5)
embeddings = get_emb_n2v(n2vmodel_hdbscan, scaled=False, pca_scaled=True)
print("Embeddings generated")

# # Save embeddings
# np.save('../data/embeddings/{embedding_method}_{model}.npy', embeddings)

# # Load embeddings
# embeddings = np.load('../data/embeddings/{embedding_method}_{model}.npy')

clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=2, metric='euclidean')
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

# Convert HDBSCAN output to fraud prediction:
# We'll treat noise points (-1) as fraud (1), and others as normal (0)
y_pred_mapped = np.where(y_pred_labelled == -1, 1, 0)

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

# Visualization

# Plot Confusion matrix
plot_confusion_matrix(x_known.numpy(), y_labelled, y_pred_labelled, title=f"{embedding_method} and {model} Confusion Matrix")

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
