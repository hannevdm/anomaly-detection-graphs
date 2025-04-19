from sklearn.metrics import classification_report
from src.data_utils import e_load, timer
from models.isolation_forest import IsolationForestModel
from sklearn.decomposition import PCA
import numpy as np
from src.embeddings import get_dgi_embeddings, train_dgi, create_dgi_model
from src.visualization import plot_pca_embeddings, plot_anomaly_scores_vs_predicted_fraud, \
    plot_confusion_matrix, plot_scores

timer(True, 'GraphSAGE and isolation forest')

# Load data
data = e_load()[0]
x = data.x

# Initialize the model
model = IsolationForestModel()
print("model initialized")

# Generate embeddings for ALL nodes
print("Generating embeddings ...")
dgi_model = create_dgi_model(in_channels=data.x.size(1), hidden_channels=32, out_channels=32)
dgi_model = train_dgi(dgi_model, data, epochs=150)
embeddings = get_dgi_embeddings(dgi_model, data, scaled=False, pca_scaled=True)
print("embeddings generated")

# # Save embeddings
# np.save('../data/embeddings/graphsageDGI.npy', embeddings)

# # Load embeddings
# embeddings = np.load('../data/embeddings/graphsageDGI.npy')


# Filter for labelled data
mask_labelled = data.y != 2
#Get the node indices of labelled nodes
labelled_indices = mask_labelled.nonzero(as_tuple=True)[0]

# Get the labels of those nodes
y_labelled = data.y[labelled_indices]

# Reduce embeddings for visualization (using ALL embeddings)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Train Isolation Forest model
print("training model ...")
model.fit(embeddings)
print("model trained")

# Predict
y_pred = model.predict(embeddings)
y_pred_known = model.filter_out_label_2(data, y_pred)
x_known = model.filter_out_label_2(data, x)
y_labelled = model.filter_out_label_2(data, data.y)

# Evaluation
accuracy, precision, recall, f1, pr_auc = model.evaluate_model(y_labelled, y_pred_known)

print(f"GraphSAGE + IF Accuracy: {accuracy}")
print(f"GraphSAGE + IF Precision: {precision}")
print(f"GraphSAGE + IF Recall: {recall}")
print(f"GraphSAGE + IF F1 Score: {f1}")
print(f"GraphSAGE + IF PR-AUC: {pr_auc}")

# Confusion Matrix
print('---- CLASSIFICATION REPORT - IF - GS ----')
y_pred_mapped = np.where(y_pred_known == -1, 1, 0)
print(classification_report(y_labelled, y_pred_mapped))

timer(False, 'GraphSAGE and isolation forest')


# Visualizations

# Plot embeddings
plot_pca_embeddings(reduced_embeddings, y_labelled)

scores = model.model.decision_function(embeddings)
scores_known = scores[labelled_indices]
fraud_scores = scores[y_pred == 1]
normal_scores = scores[y_pred == 0]

print(f"Fraud Mean Score: {np.mean(fraud_scores):.4f}")
print(f"Normal Mean Score: {np.mean(normal_scores):.4f}")


# Plot anomaly scores vs predicted fraud
plot_anomaly_scores_vs_predicted_fraud(fraud_scores, normal_scores)

# Plot anomaly scores vs fraud of labelled data
plot_scores(scores_known, y_labelled, title='IF (GraphSAGE)')

# Plot Confusion matrix
plot_confusion_matrix(x_known.numpy(), y_labelled, y_pred_known, title="GraphSAGE + IF Confusion Matrix")





