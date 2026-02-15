import cv2
from superpixels import compute_superpixels, compute_features
from skimage.color import rgb2lab

image = cv2.imread("./Test Images/simple_400.jpg")
if image is None:
  print("Input not found")
  exit()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# OpenCV takes input in BGR format so we are converting it to RGB.

lab_image = rgb2lab(image)
# Converting image to LAB(L (Lighting), A channel(green, red), B channel(blue, yellow)) for uniform perception for human eye.)

segments, num_nodes = compute_superpixels(lab_image, 300)
features = compute_features(lab_image, segments, num_nodes)

print("Number of superpixels:", num_nodes)
print(features.shape)