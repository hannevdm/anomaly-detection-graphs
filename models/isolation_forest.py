from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, \
    average_precision_score


class IsolationForestModel:

    def __init__(self, contamination=0.1, embedding_dim=32, walk_length=15, walks_per_node=5, test_size=0.2):
        """
        Constructor of the class
        :param contamination: the expected percentage of outliers in your dataset -> IF uses this to decide threshold to classify anomalies
        :param embedding_dim: the dimensionality of the node embeddings that will be learned by the Node2Vec model (each node is repr. by vector of 32 numbers)
        :param walk_length: the length of the random walks performed during the Node2Vec embedding generation -> longer walks can capture higher-order relationships in the graph
        :param walks_per_node: how many random walks will start from each node -> more walks can lead to more robust embeddings
        :param test_size: percentage of your data that will be used for testing the model after training
        """
        self.contamination = contamination
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.test_size = test_size
        self.model = IsolationForest(
            n_estimators=1000,
            contamination=self.contamination,
            max_samples=256,
            random_state=42,
            n_jobs=-1  # Use all available CPU cores for faster training
        )

    def fit(self, x_train):
        """Train Isolation Forest model with custom n_estimators."""
        self.model.fit(x_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def fit_and_predict(self, x):
        y_pred = self.model.fit_predict(x)
        return y_pred

    def evaluate_model(self, y_test, y_pred):
        """Evaluate model using precision and recall."""
        anomaly_labels = np.where(y_pred == -1, 1, 0)  # Convert to binary anomaly labels
        accuracy = accuracy_score(y_test, anomaly_labels)
        precision = precision_score(y_test, anomaly_labels)
        recall = recall_score(y_test, anomaly_labels)
        f1 = f1_score(y_test, anomaly_labels)
        pr_auc = average_precision_score(y_test, y_pred)
        return accuracy, precision, recall, f1, pr_auc

    # Utility to filter out label 2 (Unknown class)
    def filter_out_label_2(self, data, x):
        mask = data.y != 2
        return x[mask.numpy()]

