from superpixels import compute_superpixels, compute_features
from skimage.color import rgb2lab
from graph import build_adjacency, compute_weight_matrix, compute_degree_matrix, compute_normalized_similarity
from ranking import  compute_boundary_seeds, manifold_ranking
import numpy as np
import cv2

def compute_saliency(image, num_segments):
  lab_image = rgb2lab(image)
  # Converting image to LAB(L (Lighting), A channel(green, red), B channel(blue, yellow)) for uniform perception for human eye.)

  segments, num_nodes = compute_superpixels(image, num_segments)
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
  features = compute_features(lab_image, segments, num_nodes)

  variance = np.zeros(num_nodes)

  for i in range(num_nodes):
      mask = (segments == i)
      region_pixels = gray_blur[mask]
      variance[i] = np.var(region_pixels)
  variance = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)
  variance = variance * 30
  features = np.hstack((
    features[:, :3],      # L, a, b
    variance[:, None],    # texture variance
    features[:, 3:5]      # x, y
  ))

  # print(features[:, :3].max(), features[:, :3].min())
  adjacency = build_adjacency(segments, num_nodes)

  sigma = 1.81
  beta = 0.9
  weights = compute_weight_matrix(features, adjacency, sigma)
  degree = compute_degree_matrix(weights)
  normalized_similarity = compute_normalized_similarity(weights, degree)
  # Background seeds
  y_background = compute_boundary_seeds(segments, num_nodes)
  f_b = manifold_ranking(normalized_similarity, y_background, beta)
  f_b = (f_b - f_b.min()) / (f_b.max() - f_b.min())
  saliency_initial = 1 - f_b

  k = int(0.15 * num_nodes)  # 15%
  indices = np.argsort(saliency_initial)[-k:]
  y_foreground = np.zeros(num_nodes)
  y_foreground[indices] = 1

  for _ in range(2):
      f_f = manifold_ranking(normalized_similarity, y_foreground, beta)
      f_f = (f_f - f_f.min()) / (f_f.max() - f_f.min())

      saliency = f_f / (f_f + f_b + 1e-8)

      indices = np.argsort(saliency)[-k:]
      y_foreground = np.zeros(num_nodes)
      y_foreground[indices] = 1


  saliency = saliency ** 2
  saliency_map = saliency[segments]
  saliency_map = saliency_map.astype(np.float32)

  return saliency_map