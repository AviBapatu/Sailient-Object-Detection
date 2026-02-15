import cv2
from superpixels import compute_superpixels, compute_features
from skimage.color import rgb2lab
from graph import build_adjacency, compute_weight_matrix, compute_degree_matrix, compute_normalized_similarity
from ranking import  compute_boundary_seeds, manifold_ranking
import numpy as np

image = cv2.imread("./Test Images/simple_400.jpg")
if image is None:
  print("Input not found")
  exit()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# OpenCV takes input in BGR format so we are converting it to RGB.

lab_image = rgb2lab(image)
# Converting image to LAB(L (Lighting), A channel(green, red), B channel(blue, yellow)) for uniform perception for human eye.)

segments, num_nodes = compute_superpixels(image, 500)
features = compute_features(lab_image, segments, num_nodes)
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

# Foreground seeds (center superpixels)
x_coords = features[:, 3]
y_coords = features[:, 4]

dist_center = np.sqrt((x_coords - 0.5)**2 + (y_coords - 0.5)**2)

k = int(0.15 * num_nodes)  # 10%
indices = np.argsort(dist_center)[:k]

y_foreground = np.zeros(num_nodes)
y_foreground[indices] = 1

f_f = manifold_ranking(normalized_similarity, y_foreground, beta)
f_f = (f_f - f_f.min()) / (f_f.max() - f_f.min())

saliency = f_f / (f_f + f_b + 1e-8)

saliency = saliency ** 2
saliency_map = saliency[segments]

saliency_map = saliency_map.astype(np.float32)
saliency_map = cv2.bilateralFilter(
    saliency_map,
    d=9,
    sigmaColor=0.1,
    sigmaSpace=15
)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

gray_float = gray.astype(np.float32) / 255.0
image_float = image.astype(np.float32) / 255.0

saliency_3c = np.stack([saliency_map]*3, axis=-1)
output_float = saliency_3c * image_float + (1 - saliency_3c) * gray_float

output = (output_float * 255).astype(np.uint8)
cv2.imwrite("saliency.png", (saliency_map * 255).astype(np.uint8))
cv2.imwrite("output.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

# print("Number of superpixels:", num_nodes)
# print(features.shape)
# print(adjacency.shape)
# print("Total Edges:", adjacency.sum() // 2)
# print("W shape:", weights.shape)
# print("Non-zero weights:", np.count_nonzero(weights))
# print("Degree shape:", degree.shape)
# print("First 5 degrees:", degree[:5])
# print("S shape:", normalized_similarity.shape)
# print("Max value in S:", normalized_similarity.max())
# print("Min value in S:", normalized_similarity.min())
# print("y shape:", y.shape)
# print("Number of boundary seeds:", int(y.sum()))
# print("f shape:", f.shape)
# print("Min f:", f.min())
# print("Max f:", f.max())
# print("Saliency min:", saliency.min())
# print("Saliency max:", saliency.max())
# print("Saliency map shape:", saliency_map.shape)