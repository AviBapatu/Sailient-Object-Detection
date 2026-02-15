import cv2
import numpy as np
from saliency import compute_saliency

image = cv2.imread("./Test Images/realistic_400.jpg")
if image is None:
  print("Input not found")
  exit()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# OpenCV takes input in BGR format so we are converting it to RGB.

saliency_map = compute_saliency(image, 500)

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