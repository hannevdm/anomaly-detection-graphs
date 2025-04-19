from sklearn.decomposition import PCA
from models.lof import LOFModel
from src.embeddings import get_emb_gs, train_gs, GraphSAGEModel
from src.data_utils import e_load, timer
from src.visualization import plot_pca_embeddings, plot_confusion_matrix, plot_scores
timer(True, "LOF + GraphSAGE")

# Load dataset
data = e_load()[0]
x = data.x

# Initialize LOF model
lof_model = LOFModel(neighbours=30, contamination=0.1)

# GraphSAGE
print("Generating embeddings ...")
gsmodel_lof = GraphSAGEModel(in_channels= data.x.size(1), out_channels=64)
train_gs(gsmodel_lof,data)
embeddings = get_emb_gs(gsmodel_lof,data,scaled=False, pca_scaled=True)
print("embeddings generated")

# Apply LOF for anomaly detection on the embeddings
y_pred = lof_model.fit_and_predict(embeddings)

# Filter out class 2 (Unknown) from the dataset & Map test_labels_known from [0, 1] to [-1, 1]
y_pred_known = lof_model.filter_out_label_2(data, y_pred)
x_known = lof_model.filter_out_label_2(data, x)
y_labelled = lof_model.filter_out_label_2(data, data.y)
y_labelled_mapped = lof_model.change_labels(y_labelled)

# Evaluation
accuracy, precision, recall, f1, pr_auc = lof_model.evaluate(y_labelled_mapped, y_pred_known)

print(f"GraphSAGE + LOF Accuracy: {accuracy}")
print(f"GraphSAGE + LOF Precision: {precision}")
print(f"GraphSAGE + LOF Recall: {recall}")
print(f"GraphSAGE + LOF F1 Score: {f1}")
print(f"GraphSAGE + LOF PR-AUC: {pr_auc}")

timer(False, "LOF + GraphSAGE")


# Visualization

# Plot embeddings
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
plot_pca_embeddings(reduced_embeddings, y_labelled)

# Plot LOF scores of labelled data only
lof_scores = -lof_model.negative_outlier_factor_
y = data.y.cpu().numpy()
mask_labelled = y != 2
labelled_indices = mask_labelled.nonzero(as_tuple=True)[0]
lof_scores_known = lof_scores[labelled_indices]
plot_scores(lof_scores_known, y_labelled, title='LOF (GraphSAGE)')

# Plot Confusion Matrix
plot_confusion_matrix(x_known.numpy(), y_labelled, y_pred_known, title="GraphSAGE + LOF Confusion Matrix")

