from models.lof import LOFModel
from src.data_utils import e_load, timer
from src.visualization import plot_confusion_matrix, plot_scores

timer(True, "LOF + Raw")

# Load dataset
data = e_load()[0]
x = data.x

# Initialize LOF model
lof_model = LOFModel(neighbours=30, contamination=0.1)

# Apply LOF for anomaly detection on the data
y_pred = lof_model.fit_and_predict(x)

# Filter out class 2 (Unknown) from the dataset & Map test_labels_known from [0, 1] to [-1, 1]
y_pred_known = lof_model.filter_out_label_2(data, y_pred)
x_known = lof_model.filter_out_label_2(data, x)
y_labelled = lof_model.filter_out_label_2(data, data.y)
y_labelled_mapped = lof_model.change_labels(y_labelled)

# Evaluation
accuracy, precision, recall, f1, pr_auc = lof_model.evaluate(y_labelled_mapped, y_pred_known)

print(f"Raw + LOF Accuracy: {accuracy}")
print(f"Raw + LOF Precision: {precision}")
print(f"Raw + LOF Recall: {recall}")
print(f"Raw + LOF F1 Score: {f1}")
print(f"Raw + LOF PR-AUC: {pr_auc}")

timer(False, "LOF + Raw")



# Visualization

# Plot LOF scores of labelled data only
lof_scores = -lof_model.negative_outlier_factor_
y = data.y.cpu().numpy()
mask_labelled = y != 2
labelled_indices = mask_labelled.nonzero(as_tuple=True)[0]
lof_scores_known = lof_scores[labelled_indices]
plot_scores(lof_scores_known, y_labelled, title='LOF (Node Features)')

# Plot Confusion Matrix
plot_confusion_matrix(x_known.numpy(), y_labelled, y_pred_known, title="Raw + LOF Confusion Matrix")


