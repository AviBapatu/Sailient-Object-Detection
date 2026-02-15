import numpy as np
from skimage.segmentation import slic

def compute_superpixels(image, num_segments):
    segments = slic(
        image,
        n_segments = num_segments,
        compactness = 5,
        start_label = 0
    )

    num_nodes = segments.max()+1 # No. of superpixels are called num_nodes because each superpixel is called node in graph terminology.
    return segments, num_nodes

def compute_features(lab_image, segments, num_nodes):
  h, w, _ = lab_image.shape

  features = np.zeros((num_nodes, 5))

  for i in range(num_nodes):
      mask = (segments == i)
      # each pixels has a id that says which superpixel it belongs.
      # so we will be making a mask that says true or false which describes "to which superpixel the current pixel belongs to".

      region_pixels = lab_image[mask]
      # This is called boolean indexing in NumPy.
      # Here are taking every pixel that has true in boolean mask. We are transforming image space to region space.

      mean_lab = region_pixels.mean(axis=0)
      # Calculating the mean color of the total region. axis=0 represents the mean should be done column wise.
      # Mean should L1+L2..., a1+a2..., b1+b2... and output is of the size (3,_).

      rows, cols = np.where(mask)

      mean_rows = rows.mean() / h
      mean_cols = cols.mean() / w

      features[i] = [
          mean_lab[0],
          mean_lab[1],
          mean_lab[2],
          mean_rows,
          mean_cols
      ]

  return features