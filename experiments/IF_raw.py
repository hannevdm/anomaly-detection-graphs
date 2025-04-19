from models.isolation_forest import IsolationForestModel
from sklearn.metrics import classification_report
import numpy as np
from src.data_utils import e_load
from src.visualization import plot_scores, plot_anomaly_scores_vs_predicted_fraud

# Load data
dataset = e_load()
data = dataset[0]

x = data.x.cpu().numpy()
y = data.y.cpu().numpy()

# Fit IF
clf = IsolationForestModel()
clf.fit_and_predict(x)

# Predict
y_pred = clf.predict(x)
y_pred = np.where(y_pred == 1, 0, 1)  # 1 → inliers (licit), -1 → fraud

# Filter
mask_labelled = y != 2
y_labelled = y[mask_labelled]
y_pred_labelled = y_pred[mask_labelled]


# Evaluation
accuracy, precision, recall, f1, pr_auc = clf.evaluate_model(y_labelled, y_pred_labelled)

print(f"Node2Vec + LOF Accuracy: {accuracy}")
print(f"Node2Vec + LOF Precision: {precision}")
print(f"Node2Vec + LOF Recall: {recall}")
print(f"Node2Vec + LOF F1 Score: {f1}")
print(f"Node2Vec + LOF PR-AUC: {pr_auc}")

print('---- CLASSIFICATION REPORT - IF - RAW ----')
print(classification_report(y_labelled, y_pred_labelled))


# Plot anomaly scores of labelled data only
scores = clf.model.decision_function(x)
labelled_indices = mask_labelled.nonzero(as_tuple=True)[0]
scores_known = scores[labelled_indices]
plot_scores(scores_known, y_labelled, title='IF (Node Features)')

# Plot anomaly scores vs predicted fraud
fraud_scores = scores[y_labelled == 1]
normal_scores = scores[y_labelled == 0]
plot_anomaly_scores_vs_predicted_fraud(fraud_scores, normal_scores)

# Plot Confusion matrix
