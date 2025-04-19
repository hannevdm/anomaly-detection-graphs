import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.data_utils import e_load, e_drop_labels, timer
from src.visualization import plot_loss, plot_pca_embeddings, plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, \
    average_precision_score
from models.vgae import VGAEModel

timer(True, "VGAE")

# Load and preprocess data
data = e_load()
data, labels = e_drop_labels(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normalize features globally
scaler = StandardScaler()
data.x = torch.tensor(scaler.fit_transform(data.x.cpu()), dtype=torch.float32).to(device)

# Instantiate the model
in_channels = data.x.shape[1]
hidden_channels = 64
out_channels=8
model = VGAEModel(in_channels, hidden_channels, out_channels)

# Move data and model to GPU if available
model = model.to(device)
data = data.to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Run the training
epochs = 150
losses=[]
for epoch in range(epochs):
    loss = model.train_with_laplacian(optimizer, data, device)
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')
    losses.append(loss)

plot_loss(losses, name="VGAE")

# Evaluation using original labels
model.eval()
with torch.no_grad():
    mu, logstd = model.encode(data.x, data.edge_index)
    z = model.reparametrize(mu, logstd)


# ======== Anomaly Scoring ========

# KL divergence per node
kl_per_node = -0.5 * torch.sum(1 + 2 * logstd - mu ** 2 - torch.exp(2 * logstd), dim=1)

# Node reconstruction error
def node_reconstruction_error(z, edge_index):
    preds = model.decode(z, edge_index)
    errors = torch.abs(preds - 1.0)
    node_errors = torch.zeros(z.size(0)).to(device)
    counts = torch.zeros(z.size(0)).to(device)
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i], edge_index[1, i]
        node_errors[u] += errors[i]
        node_errors[v] += errors[i]
        counts[u] += 1
        counts[v] += 1
    return (node_errors / counts.clamp(min=1))

recon_error_per_node = node_reconstruction_error(z, data.edge_index)

# Combine both scores
anomaly_scores = 0.7 * kl_per_node + 0.3 * recon_error_per_node
anomaly_scores = anomaly_scores.cpu().numpy()

# Evaluation metrics (fraud = 1, licit = 0)
true_labels = labels.cpu().numpy()
mask = true_labels != 2
true_labels_filtered = true_labels[mask]
anomaly_scores_filtered = anomaly_scores[mask]

# Setting threshold based on F1 precision and recall
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
# --- Precision-Recall Curve ---
precisions, recalls, thresholds = precision_recall_curve(true_labels_filtered, anomaly_scores_filtered, pos_label=1)
plt.figure(figsize=(12, 5))
# Precision-Recall vs Threshold
plt.subplot(1, 2, 1)
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Threshold")
plt.legend()
# Anomaly Score Distribution
plt.subplot(1, 2, 2)
plt.hist(anomaly_scores_filtered[true_labels_filtered == 0], bins=50, alpha=0.6, label="Licit")
plt.hist(anomaly_scores_filtered[true_labels_filtered == 1], bins=50, alpha=0.6, label="Fraud")
plt.axvline(np.percentile(anomaly_scores_filtered, 80), color='r', linestyle='--', label='80th percentile')
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.title("Anomaly Score Distribution")
plt.legend()
plt.tight_layout()
plt.show()
# --- User Input for Threshold ---
while True:
    try:
        user_input = input("Enter your desired threshold: ")
        threshold = float(user_input)
        break
    except ValueError:
        print("Invalid input. Please enter a valid number.")
# Apply threshold
binary_preds = (anomaly_scores_filtered >= threshold).astype(int)

accuracy = accuracy_score(true_labels_filtered, binary_preds)
precision = precision_score(true_labels_filtered, binary_preds, pos_label=1)
recall = recall_score(true_labels_filtered, binary_preds, pos_label=1)
f1 = f1_score(true_labels_filtered, binary_preds, pos_label=1)
pr_auc = average_precision_score(true_labels_filtered, binary_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"PR-AUC-score: {pr_auc:.4f}")

timer(False, "VGAE")

print('---- CLASSIFICATION REPORT - VGAE ----')
print(classification_report(true_labels_filtered, np.array(binary_preds)))


# ================= Visualizations for VGAE =================

# Reduce to 2D with PCA for embedding visualization
z_cpu = z.cpu().detach().numpy()
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(z_cpu)

# Plot PCA embeddings
plot_pca_embeddings(reduced_embeddings, true_labels_filtered)

# Histogram of anomaly scores of labelled data only
plt.hist(anomaly_scores_filtered[true_labels_filtered == 0], bins=50, alpha=0.6, label="Licit")
plt.hist(anomaly_scores_filtered[true_labels_filtered == 1], bins=50, alpha=0.6, label="Fraud")
plt.axvline(np.percentile(anomaly_scores_filtered, 80), color='r', linestyle='--', label='80th percentile')
plt.axvline(threshold, color='b', linestyle='--', label='chosen threshold')
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.title("VGAE Anomaly Score Distribution")
plt.legend()
plt.show()

# Plot confusion matrix with UMAP projection
plot_confusion_matrix(z_cpu[mask], true_labels_filtered, np.array(binary_preds), title="UMAP + VGAE Predictions", map=False)
