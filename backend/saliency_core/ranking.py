import numpy as np
from scipy.linalg import solve

def compute_boundary_seeds(segments, num_nodes):
  h, w = segments.shape

  boundary_ids = set()

  # Top Row
  boundary_ids.update(segments[0, :])
  # Bottom Row
  boundary_ids.update(segments[h-1, :])
  # Left Col
  boundary_ids.update(segments[:, 0])
  # Right Col
  boundary_ids.update(segments[:, w-1])

  y = np.zeros(num_nodes)
  for idx in boundary_ids:
    y[idx] = 1

  return y

def manifold_ranking(S, y, beta):
    num_nodes = S.shape[0]
    I = np.eye(num_nodes)
    A = I - beta * S

    f = solve(A, y)
    return f