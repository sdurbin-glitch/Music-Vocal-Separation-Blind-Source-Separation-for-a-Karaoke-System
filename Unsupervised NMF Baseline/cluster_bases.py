import numpy as np


def cluster_bases(B, W, n_clusters, max_iter=100):
    if B.ndim != 2 or W.ndim != 2 or B.shape[1] != W.shape[0]:
        raise ValueError("B and W must be 2D arrays with matching component dimensions.")
    if not 1 <= n_clusters <= B.shape[1]:
        raise ValueError("n_clusters must be between 1 and the number of components.")

    # Normalize each component's spectrum and activation trajectory separately.
    spectral_features = B.T
    spectral_features = spectral_features / np.maximum(np.linalg.norm(spectral_features, axis=1, keepdims=True), 1e-10)
    activation_features = W
    activation_features = activation_features / np.maximum(np.linalg.norm(activation_features, axis=1, keepdims=True), 1e-10)

    # Equalize the two feature blocks so neither dominates by dimensionality.
    features = np.concatenate((spectral_features, activation_features), axis=1) / np.sqrt(2)

    # Pick diverse seed centers using farthest-point initialization.
    distances = np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=2)
    first_center = int(np.argmax(distances.sum(axis=1)))
    centers = [features[first_center]]
    chosen = {first_center}
    while len(centers) < n_clusters:
        squared_distances = np.sum((features[:, None, :] - np.stack(centers)[None, :, :]) ** 2, axis=2)
        min_distances = squared_distances.min(axis=1)
        next_center = int(np.argmax(min_distances))
        if next_center in chosen:
            # If a duplicate candidate is selected, use the furthest remaining feature.
            candidate_order = np.argsort(min_distances)[::-1]
            next_center = int(candidate_order[np.isin(candidate_order, list(chosen), invert=True)][0])
        chosen.add(next_center)
        centers.append(features[next_center])
    centers = np.stack(centers)

    labels = np.zeros(B.shape[1], dtype=int)
    for _ in range(max_iter):
        squared_distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(squared_distances, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for cluster in range(n_clusters):
            members = features[labels == cluster]
            if members.size:
                centers[cluster] = members.mean(axis=0)
            else:
                centers[cluster] = features[np.argmax(squared_distances.min(axis=1))]
    return labels
