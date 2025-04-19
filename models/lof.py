import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.neighbors import LocalOutlierFactor


class LOFModel:
    def __init__(self, neighbours = 30, contamination = 0.02, novelty=False):
        self.neighbours = neighbours
        self.contamination = contamination
        self.novelty = novelty
        self.lof = LocalOutlierFactor(n_neighbors=self.neighbours, contamination=self.contamination, novelty= self.novelty)
        self.negative_outlier_factor_ = None

    def train_test(self, data):
        train_mask = data.train_mask
        test_mask = data.test_mask
        return train_mask, test_mask

    def fit_and_predict(self, x):
        y_pred = self.lof.fit_predict(x)
        self.negative_outlier_factor_ = self.lof.negative_outlier_factor_
        return y_pred

    def predict(self, x_test):
        return self.lof.predict(x_test)

    def fit(self, x_train):
        self.lof.fit(x_train)

    def change_labels(self, labels):
        # Map test_labels from [0, 1] to [-1, 1]
        labels = np.where(labels == 0, 1, -1)
        return labels

    def filter_out_label_2(self,data, x):
        mask = data.y != 2
        return x[mask.numpy()]

    def evaluate(self, test_labels_known, y_pred_known):
        accuracy = accuracy_score(test_labels_known,y_pred_known)
        precision = precision_score(test_labels_known, y_pred_known, pos_label=-1)
        recall = recall_score(test_labels_known, y_pred_known, pos_label=-1)
        f1 = f1_score(test_labels_known, y_pred_known, pos_label=-1)
        pr_auc = average_precision_score(test_labels_known, y_pred_known)
        return accuracy, precision, recall, f1, pr_auc