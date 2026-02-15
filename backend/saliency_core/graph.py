import numpy as np

def build_adjacency(segments, num_nodes):
  h, w = segments.shape # Getting height, width to iterate through all the pixels
  adjacency = np.zeros((num_nodes, num_nodes), dtype=np.uint8)
  # Initializing adjacency matrix with size of num_nodesxnum_nodes with type uint8 because we are saving only 0 or 1 so it saves memory and faster than float.

  for r in range(h):
    for c in range(w):
      current_pos = segments[r][c]
      # We are only taking right, bottom neighbours and not all the four neighbours because if we do there are going to be duplicate
      # Take this example
      # A B
      # C D
      # A is left neighbour to B and a top neighbour to C, also B is right neighbour to A and C is bottom neighbour to A. By taking only right and bottom we are eliminating the duplicate computation.

      # Check right neighbour
      if c+1 < w:
        right = segments[r, c+1]
        if current_pos != right:
          adjacency[current_pos, right] = 1
          adjacency[right, current_pos] = 1

      # Check bottom neighbour
      if r+1 < h:
        bottom = segments[r+1, c]
        if current_pos != bottom:
          adjacency[current_pos, bottom] = 1
          adjacency[bottom, current_pos] = 1
  return adjacency


def compute_weight_matrix(features, adjacency, sigma):
  num_nodes = features.shape[0] # Taking the size of the features matrix
  weights = np.zeros((num_nodes, num_nodes)) # Initializing a matrix of size num_nodesxnum_nodes

  # Iterating through nodes in such a way that duplicate iteration is not computed
  # through only j > i and then assiging the same weight for weights[i][j] and weights[j][i].

  for i in range(num_nodes):
    for j in range(i+1, num_nodes):
      if adjacency[i][j] == 1:
        diff = features[i] - features[j]
        dist_sq = np.sum(diff ** 2)

        weight = np.exp(-dist_sq / (2 * sigma * sigma)) # Gaussian Kernel
        weights[i][j] = weight
        weights[j][i] = weight
  return weights

def compute_degree_matrix(weights):
  degree = np.sum(weights, axis=1)
  return degree

def compute_normalized_similarity(weights, degree):
    degree_inv_sqrt = 1.0 / np.sqrt(degree)
    normalizded_similarity = weights * degree_inv_sqrt[:, None] * degree_inv_sqrt[None, :]
    return normalizded_similarity
