# import
from skimage import feature

# extract features from the image in the 
# form of HOG
def quantify_image(image):

	features = feature.hog(image, orientations=9,
		pixels_per_cell=(10, 10), cells_per_block=(2, 2),
		transform_sqrt=True, block_norm="L1")

	return features