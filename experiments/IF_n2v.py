from sklearn.metrics import classification_report
from src.data_utils import e_load, timer
from models.isolation_forest import IsolationForestModel
from sklearn.decomposition import PCA
import numpy as np
from src.embeddings import Node2VecModel, train_n2v, get_emb_n2v
from src.visualization import plot_pca_embeddings, plot_anomaly_scores_vs_predicted_fraud, \
    plot_confusion_matrix, plot_scores

timer(True, 'N2V and isolation forest')

# Load data
data = e_load()[0]
x = data.x

# Generate embeddings
print("Generating embeddings ...")
n2vmodel_if = Node2VecModel(data.edge_index, embedding_dim=128, walk_length=120, context_size=10, walks_per_node=100,num_negative_samples=1,
                            p=2, q=0.5)
train_n2v(n2vmodel_if, 5)
embeddings = get_emb_n2v(n2vmodel_if, scaled=False, pca_scaled=True)
print("embeddings generated")

# Save embeddings
# np.save('../data/embeddings/node2vec.npy', embeddings)

# Load embeddings
# embeddings = np.load('../data/embeddings/node2vec.npy')

# Initialize the model
model = IsolationForestModel()
print("model initialized")

# Train Isolation Forest model
print("training model ...")
model.fit(embeddings)
print("model trained")

# Predict
y_pred = model.predict(embeddings)

# Filter for labelled data
mask_labelled = data.y != 2
#Get the node indices of labelled nodes
labelled_indices = mask_labelled.nonzero(as_tuple=True)[0]

# Get the labels of those nodes
y_labelled = data.y[labelled_indices]
y_pred_known = y_pred[labelled_indices]
y_pred_mapped_known = np.where(y_pred_known == -1, 1, 0)
x_known = model.filter_out_label_2(data, x)


# Evaluation
accuracy, precision, recall, f1, pr_auc = model.evaluate_model(y_labelled, y_pred_known)

print(f"Node2Vec + IF Accuracy: {accuracy}")
print(f"Node2Vec + IF Precision: {precision}")
print(f"Node2Vec + IF Recall: {recall}")
print(f"Node2Vec + IF F1 Score: {f1}")
print(f"Node2Vec + IF PR-AUC: {pr_auc}")

print('---- CLASSIFICATION REPORT - IF - N2V ----')
print(classification_report(y_labelled, y_pred_mapped_known))

timer(False, 'N2V and isolation forest')


# Visualisations

# Reduce embeddings for visualization (using ALL embeddings)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
plot_pca_embeddings(reduced_embeddings, y_labelled)

scores = model.model.decision_function(embeddings)
scores_known = scores[labelled_indices]
fraud_scores = scores[y_labelled == 1]
normal_scores = scores[y_labelled == 0]

print(f"Fraud Mean Score: {np.mean(fraud_scores):.4f}")
print(f"Normal Mean Score: {np.mean(normal_scores):.4f}")


# Plot anomaly scores vs predicted fraud
plot_anomaly_scores_vs_predicted_fraud(fraud_scores, normal_scores)

# Plot anomaly scores of labelled data only
plot_scores(scores_known, y_labelled, title='IF (Node2Vec)')

# Plot Confusion matrix
plot_confusion_matrix(x_known.numpy(), y_labelled, y_pred_known, title="Node2Vec + IF Confusion Matrix")

