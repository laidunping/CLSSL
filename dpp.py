import numpy as np
import math

def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s = di2s - np.square(eis)
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def get_dpp(features, sample_size, labels, num_class):
    max_length = sample_size
    item_size = int(len(features))
    centers = class_centers(features, labels, num_class)
    # print(centers.shape)
    centers = expand_centers(centers, labels)
    scores = np.reciprocal(np.linalg.norm(features - centers, axis=1))
    scores = np.exp(0.1 * scores)
    #print(scores)
    # scores = np.exp(0.01 * np.ones(item_size) + 0.2)
    feature_vectors = features
    feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    similarities = np.dot(feature_vectors, feature_vectors.T)
    #print(np.max(similarities), np.min(similarities))
    kernel_matrix = scores.reshape((item_size, 1)) * similarities * scores.reshape((1, item_size))

    sampled_indices = dpp(kernel_matrix, max_length)

    return sampled_indices

def class_centers(data, target, num_classes):
    # Initialize a list to store the sample index for each category
    class_indices = [[] for _ in range(num_classes)]

    # Traverse each sample and add its index to the list of corresponding categories
    for i in range(target.shape[0]):
        class_indices[target[i]].append(i)


    class_centers = np.zeros((num_classes, data.shape[1]))

    # Calculate the category center for each category
    for class_idx in range(num_classes):
        if class_indices[class_idx]:
            class_data = data[class_indices[class_idx]]
            class_centers[class_idx] = np.mean(class_data, axis=0)

    return class_centers

def expand_centers(centers, target):
    # Use label index to broadcast the category center to the corresponding sample location
    expanded_centers = centers[target]
    return expanded_centers