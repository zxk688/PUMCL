import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as col

def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              source_labels: torch.Tensor, target_labels: torch.Tensor,
              filename: str):
    """
    Visualize features from different domains using t-SNE with different shapes and colors for each class.

    Args:
        source_feature (torch.Tensor): features from source domain in shape (minibatch, F)
        target_feature (torch.Tensor): features from target domain in shape (minibatch, F)
        source_labels (torch.Tensor): labels of source domain samples
        target_labels (torch.Tensor): labels of target domain samples
        filename (str): the file name to save t-SNE plot
    """

    # Convert tensors to numpy arrays
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    source_labels = source_labels.numpy()
    target_labels = target_labels.numpy()

    # Determine the number of classes and unique labels
    unique_source_labels = np.unique(source_labels)
    unique_target_labels = np.unique(target_labels)
    num_source_classes = len(unique_source_labels)
    num_target_classes = len(unique_target_labels)

    # Map features to 2D using t-SNE
    features = np.concatenate([source_feature, target_feature], axis=0)
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # Define colors for each class
    class_colors = ["#33ccff","#33ffff", "#33ffaa", "#33ff33", "#ccff33", "#ffff33","#ffcc22","#ff3333"]
   
    plt.figure(figsize=(10, 10))

    # Plot source domain samples
    for i, label in enumerate(unique_source_labels):
        source_labels=source_labels.flatten()
        idx = source_labels == label
        label_features = X_tsne[:len(source_labels)][idx]
        class_color = class_colors[i % num_source_classes]
        plt.scatter(label_features[:, 0], label_features[:, 1],
                    c=class_color, marker='o')

    # Plot target domain samples
    for i, label in enumerate(unique_target_labels):
        target_labels=target_labels.flatten()
        idx = target_labels == label
        label_features = X_tsne[len(source_labels):][idx]
        class_color = class_colors[i % num_target_classes]
        plt.scatter(label_features[:, 0], label_features[:, 1],
                    c=class_color, marker='^')

    plt.legend(loc='best')
    plt.savefig(filename, dpi=200, bbox_inches='tight')


