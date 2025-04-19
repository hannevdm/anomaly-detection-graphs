from models.lof import LOFModel
from src.embeddings import train_n2v, get_emb_n2v, Node2VecModel
from src.data_utils import e_load, timer
from sklearn.decomposition import PCA
from src.visualization import plot_pca_embeddings, plot_confusion_matrix, plot_scores

timer(True, "Node2Vec + LOF")

# Load the dataset
data = e_load()[0]
x = data.x

# Initialize LOF model
lof_model = LOFModel(neighbours=30, contamination=0.1)

# Node 2 Vec
print("Generating embeddings ...")
n2vmodel_lof = Node2VecModel(data.edge_index, embedding_dim=256, walk_length=120, context_size=10, walks_per_node=100, num_negative_samples=10,
                            p=2, q=0.5)
train_n2v(n2vmodel_lof, 5)
embeddings = get_emb_n2v(n2vmodel_lof, scaled=False, pca_scaled=True)
print("embeddings generated")

# Save embeddings
# np.save('../data/embeddings/node2vec.npy', embeddings)

# Load embeddings
# embeddings = np.load('../data/embeddings/node2vec.npy')

# add node features to embeddings (optional)
# nx = pytorch_to_networkx(data)
# node_features = extract_network_features(nx, data.num_nodes)
# embeddings = np.hstack([embeddings, node_features])

y_pred = lof_model.fit_and_predict(embeddings)

# Filter out class 2 (Unknown) from the dataset & Map test_labels_known from [0, 1] to [-1, 1]
y_pred_known = lof_model.filter_out_label_2(data, y_pred)
x_known = lof_model.filter_out_label_2(data, x)
y_labelled = lof_model.filter_out_label_2(data, data.y)
y_labelled_mapped = lof_model.change_labels(y_labelled)

# Evaluation
accuracy, precision, recall, f1, pr_auc = lof_model.evaluate(y_labelled_mapped, y_pred_known)

print(f"Node2Vec + LOF Accuracy: {accuracy}")
print(f"Node2Vec + LOF Precision: {precision}")
print(f"Node2Vec + LOF Recall: {recall}")
print(f"Node2Vec + LOF F1 Score: {f1}")
print(f"Node2Vec + LOF PR-AUC: {pr_auc}")

timer(False, "LOF + Node2Vec")


# Visualization

# Visualize embeddings
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
plot_pca_embeddings(reduced_embeddings, y_labelled)

# Plot LOF scores of labelled data only
lof_scores = -lof_model.negative_outlier_factor_
y = data.y.cpu().numpy()
mask_labelled = y != 2
labelled_indices = mask_labelled.nonzero(as_tuple=True)[0]
lof_scores_known = lof_scores[labelled_indices]
plot_scores(lof_scores_known, y_labelled, title='LOF (Node2Vec)')

# Confusion matrix
plot_confusion_matrix(x_known.numpy(), y_labelled, y_pred_known, title="Node2Vec + LOF Confusion Matrix")
